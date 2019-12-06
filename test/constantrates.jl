
using Distributions
using Beluga
using DataFrames
using Parameters
using TransformVariables
using ForwardDiff
using CSV
using Plots
using StatsPlots

#===============================================================================
                        Constant rates model definition
===============================================================================#

mutable struct ConstantRatesModel{T} <: ContinuousMultivariateDistribution where T<:Real
    prior::ConstantRatesPrior
    tree::SpeciesTree
    λ::T
    μ::T
    η::T
end
function ConstantRatesModel(prior::ConstantRatesPrior, tree::SpeciesTree, λ::T, μ::T, η::T) where T<:Real
    return ConstantRatesModel{T}(prior, tree, λ, μ, η)
end
function ConstantRatesModel{T}(nt::NamedTuple) where T<:Real
    return ConstantRatesModel{T}(prior, tree, nt[:λ], nt[:μ], nt[:η])
end
function ConstantRatesModel{T}(x::AbstractVector) where T<:Real
    return ConstantRatesModel{T}(prior, tree, x[1], x[2], x[3])
end

function logprior(d::ConstantRatesModel)
    λ=d.λ
    μ=d.μ
    q=0.0
    η=d.η
    @unpack dλ, dμ, dq, dη = d.prior
    lp = logpdf(dλ, λ) + logpdf(dμ, μ) + logpdf(dη, η)
    lp += sum(logpdf.(dq, q))
    return lp
end

function Distributions.loglikelihood(d::ConstantRatesModel, data::AbstractArray{T, 2}) where T<:Real
    tree = d.tree
    λ = repeat([d.λ], length(tree))
    μ = repeat([d.μ], length(tree))
    d_model = DuplicationLoss(tree, λ, μ, d.η, maximum(data))
    # println(d_model)
    return Beluga.logpdf(d_model, data)
end

function model_transform(d::ConstantRatesModel)
    t = as((λ = as_positive_real, μ = as_positive_real, η = as_unit_interval))
    return t
end

function model_invtransform(d::ConstantRatesModel)
    t = inverse(model_transform(d))
    return t
end

function update_params!(d::ConstantRatesModel, θ::NamedTuple) where T<:Real
    d.prior = ConstantRatesPrior(θ...)
end


#===============================================================================
                        Overloaded ELBO functions
===============================================================================#

function (elbo::ELBO)(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}
    r = 0.0
    N = elbo.n_samples
    n = M(length(q.dists))

    for i in 1:N
        η = rand(n)                         # Standardized parameters
        ζ = inv_elliptical(q, η)            # Real space parameters
        transform = model_transform(model)
        θ = transform(ζ)                    # Parameter space
        m = ConstantRatesModel(model.prior, model.tree, θ...)
        r += logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2]
    end

    return r/N - entropy(q)
end

function grad_elbo(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}

    # Get variational parameters (real space)
    p_real = sample_invtransform(q)(Distributions.params(q))

    # Create standard normal variational distribution
    N = length(q.dists)
    n = M(N)
    p = sample_transform(q)(p_real)     # Variational params in parameter space
    Q = M(p)
    η = rand(n)                         # Sample standard normal
    ζ = Vector{Float64}(inv_elliptical(Q, η))            # Transform to real space
    transform = model_transform(model)
    θ = [transform(ζ)...]               # Model parameters

    function grad_logprior(θ::Vector{Float64})
        f(x) = begin
            m = ConstantRatesModel(model.prior, model.tree, x[1], x[2], x[3])
            return logprior(m)
        end
        return ForwardDiff.gradient(f, θ)
    end
    function grad_loglikelihood(θ::Vector{Float64})
        m = DuplicationLoss(model.tree, [θ[1]], [θ[2]], θ[3], maximum(data))
        return Beluga.gradient(m, data)
    end
    function grad_logdetjac(ζ::Vector{Float64})
        f(x) = begin
            return TransformVariables.transform_and_logjac(transform, x)[2]
        end
        return ForwardDiff.gradient(f, ζ)
    end
    function grad_invtransform(ζ::Vector{Float64})
        return [exp(ζ[1]), exp(ζ[2]), exp(-ζ[3])]
    end
    function grad_entropy()
        return Vector{Float64}([zeros(N)..., ones(N)...])
    end

    x::Vector{Float64} = ((grad_logprior(θ) .+ grad_loglikelihood(θ)) .* grad_invtransform(ζ) .+ grad_logdetjac(ζ))
    μ::Vector{Float64} = x
    ω::Vector{Float64} = x .* (η .* p[2])
    return Vector{Float64}([μ..., ω...]) .+ grad_entropy()
end

function calc_grad_elbo(q::M, model::D, data::AbstractArray{T}, N::Int=10) where {M<:MeanField, D<:Distribution, T<:Real}

    n = dimension(sample_transform(q))
    r = zeros(n)
    for i in 1:N
        r .+= grad_elbo(q, model, data)
    end
    r ./= N
    return r
end


#===============================================================================
                            Main program  for testing
===============================================================================#

# # Get some example data
# tree, _ = Beluga.example_data1()
# Beluga.set_constantrates!(tree)
# df = DataFrame(:A=>[4,3,1,4,3],:B=>[2,1,3,2,2],:C=>[4,4,1,1,3],:D=>[3,3,1,5,2])
# data = profile(tree, df)
#
# true_params = [0.1, 0.2, 0.3, 0.8]
# λ_init, μ_init, q_init, η_init = 0.5, 0.5, 0.0, 0.5
#
# prior = ConstantRatesPrior(Exponential(1.), Exponential(1.), Beta(1., 1.), Beta(6., 2.))
# model = ConstantRatesModel{Float64}(prior, tree, λ_init, μ_init, η_init)
# Q = MeanFieldGaussian([Normal(0.1, 0.1) for i in 1:3]) # Initial mean-field distribution
#
# elbo = ELBO(50)
# elbo(Q, model, data)
# advi = ADVI(5, 100, 1, 100, VarInfLogger(Vector(), Vector(), Vector(), DataFrame(repeat([Float64[]], 13))))
# res = advi(elbo, Q, model, data, 0.1, 0.8)
# res = Distributions.params(res)[:μ]
# res = model_transform(model)(res)
# print(res)
#
# using Plots
# using StatsPlots
# using LaTeXStrings
#
# df = advi.logger.df
# labels = ["\\lambda", "\\mu", "\\eta"]
# p = []
# for i in 1:3
#     push!(p, Plots.plot(df[i], ribbon=df[i+3], title=labels[i], legend=:none))
# end
# push!(p, Plots.plot(df[end], title="ELBO", legend=:none))
# Plots.plot(p...)


#===============================================================================
                            Test VI on simulated data
===============================================================================#

# Test on simulated data
df = CSV.read(".\\PhyloVI\\data\\N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
df = CSV.read(".\\PhyloVI\\data\\N=250_tree=plants1c.nw_η=0.9_λ=5_μ=4.csv", delim=",")
# df = CSV.read(".\\PhyloVI\\data\\N=1000_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
# df = CSV.read(".\\PhyloVI\\data\\plants1.filter012.csv", delim=",")
tree = SpeciesTree(".\\PhyloVI\\data\\plants1c.nw")
Beluga.set_constantrates!(tree)
data = profile(tree, df)

prior = ConstantRatesPrior(Exponential(1.), Exponential(1.), Beta(1., 1.), Beta(6., 2.))
λ_init, μ_init, q_init, η_init = 0.5, 0.5, 0.0, 0.5
model = ConstantRatesModel{Float64}(prior, tree, λ_init, μ_init, η_init)
Q = MeanFieldGaussian([Normal(0., 1.) for i in 1:3]) # Initial mean-field distribution

elbo = ELBO(0)
advi = ADVI(1, 500, 2, 200, VarInfLogger(Vector(), Vector(), Vector(), DataFrame(repeat([Float64[]], 13))))
_, res = advi(elbo, Q, model, data, .03, 0.01, 0.1)
res = Distributions.params(res)[:μ]
res = model_transform(model)(res)
print(res)

df_res = advi.logger.df
labels = ["\\lambda", "\\mu", "\\eta"]
p = []
for i in 1:3
    push!(p, Plots.plot((df_res[i]), ribbon=df_res[i+3], title=labels[i], legend=:none))
end
push!(p, Plots.plot(df_res[end], title="ELBO", legend=:none))
Plots.plot(p...)



#===============================================================================
                        Test MCMC on simulated data
===============================================================================#
# p, m = Profile(df, tree)
# prior = ConstantRatesPrior(
#     Exponential(1), Exponential(1), Beta(1,1), Beta(6,2))
# chain2 = DLChain(p, prior, tree, m)
# chain2 = mcmc!(chain2, 1_000, show_every=100)
