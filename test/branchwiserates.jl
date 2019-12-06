
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

mutable struct ExpRatesModel{T} <: ContinuousMultivariateDistribution where T<:Real
    prior::ExpRatesPrior
    tree::SpeciesTree
    λ::Vector{T}
    μ::Vector{T}
    η::T
end
function ExpRatesModel(prior::ExpRatesPrior, tree::SpeciesTree, λ::T, μ::T, η::T) where T<:Real
    return ExpRatesModel{T}(prior, tree, λ, μ, η)
end
function ExpRatesModel{T}(nt::NamedTuple) where T<:Real
    return ExpRatesModel{T}(prior, tree, nt[:λ], nt[:μ], nt[:η])
end
function ExpRatesModel{T}(x::AbstractVector) where T<:Real
    N = Int((length(x) - 1)/2)
    return ExpRatesModel{T}(prior, tree, x[1:N], x[N+1:2N], x[end])
end

function logprior(d::ExpRatesModel)
    λ=d.λ
    μ=d.μ
    q=0
    η=d.η
    @unpack dλ, dμ, dq, dη = d.prior
    lp = logpdf(dη, η)
    lp += sum(logpdf.(dλ, λ))
    lp += sum(logpdf.(dμ, μ))
    lp += sum(logpdf.(dq, q))
    return lp
end

function Distributions.loglikelihood(d::ExpRatesModel, data::AbstractArray{T, 2}) where T<:Real
    tree = d.tree
    λ = d.λ
    μ = d.μ
    d_model = DuplicationLoss(tree, λ, μ, d.η, maximum(data))
    return Beluga.logpdf(d_model, data)
end

function model_transform(d::ExpRatesModel)
    N = length(d.tree)
    t = as((λ = as(Array, as_positive_real, N), μ = as(Array, as_positive_real, N), η = as_unit_interval))
    return t
end

function model_invtransform(d::ExpRatesModel)
    t = inverse(model_transform(d))
    return t
end

function update_params!(d::ExpRatesModel, θ::NamedTuple) where T<:Real
    d.prior = ExpRatesPrior(θ...)
end


#===============================================================================
                        Overloaded ELBO functions
===============================================================================#

function (elbo::ELBO)(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}

    N = elbo.n_samples
    if N == 0; return 0.0; end
    r = 0.0
    n = M(length(q.dists))

    for i in 1:N
        η = rand(n)                         # Standardized parameters
        ζ = inv_elliptical(q, η)            # Real space parameters
        transform = model_transform(model)
        θ = transform(ζ)                    # Parameter space
        m = ExpRatesModel(model.prior, model.tree, θ...)
        r += logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2]
    end

    return r/N - entropy(q)
end

function grad_elbo(q::M, model::D, data::AbstractArray{T}, batch_factor::Float64=1.0) where {M<:MeanField, D<:Distribution, T<:Real}

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
    θ = Float64[i for i in Iterators.flatten([transform(ζ)...])]      # Model parameters

    function grad_logprior(θ::Vector{Float64})
        f(x) = begin
            t = length(model.tree)
            m = ExpRatesModel(model.prior, model.tree, x[1:t], x[t+1:2t], x[end])
            return logprior(m)
        end
        return ForwardDiff.gradient(f, θ)
    end
    function grad_loglikelihood(θ::Vector{Float64})
        t = length(model.tree)
        m = DuplicationLoss(model.tree, θ[1:t], θ[t+1:2t], θ[end], maximum(data))
        return Beluga.gradient(m, data)
    end
    function grad_logdetjac(ζ::Vector{Float64})
        f(x) = begin
            return TransformVariables.transform_and_logjac(transform, x)[2]
        end
        return ForwardDiff.gradient(f, ζ)
    end
    function grad_invtransform(ζ::Vector{Float64})
        t = ζ[end] > 0 ? exp(-ζ[end]) : exp(ζ[end])
        return [exp.(ζ[1:end-1])..., t]
    end
    function grad_entropy()
        return Vector{Float64}([zeros(N)..., ones(N)...])
    end

    #println(length(grad_logprior(θ)), "; ", length(grad_loglikelihood(θ)), "; ", length(grad_invtransform(ζ)), "; ", length(grad_logdetjac(ζ)))
    x::Vector{Float64} = ((grad_logprior(θ) .+ grad_loglikelihood(θ)) .* grad_invtransform(ζ) .+ grad_logdetjac(ζ))
    μ::Vector{Float64} = x
    ω::Vector{Float64} = x .* (η .* p[2])
    return Vector{Float64}([μ..., ω...]) .+ grad_entropy()
end

function calc_grad_elbo(q::M, model::D, data::AbstractArray{T}, N::Int=10, batch_factor::Float64=1.0) where {M<:MeanField, D<:Distribution, T<:Real}

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

using Plots
using StatsPlots
using LaTeXStrings
using Measurements

#===============================================================================
                            Branchwise experiments
===============================================================================#

# Test on simulated data
# df = CSV.read(".\\PhyloVI\\data\\N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
#df = CSV.read(".\\PhyloVI\\data\\N=250_tree=plants1c.nw_η=0.9_λ=5_μ=4.csv", delim=",")
df = CSV.read(".\\PhyloVI\\data\\N=1000_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
#df = CSV.read(".\\PhyloVI\\data\\plants1.filter012.csv", delim=",")
df = CSV.read(".\\PhyloVI\\data\\branch_wise\\1.counts.csv", delim=",")
df = CSV.read(".\\PhyloVI\\data\\branch_wise\\N=5000_burnin=1000_cv=0.99_n=11000_pk=0.1_qa=2_qaa=1_qb=3_qbb=1_r=1.5_s=0.5_ss=0.4_ηa=5_ηb=1_σ=0.5_σ0=0.5.1.counts.csv", delim=",")
# df = CSV.read(".\\PhyloVI\\data\\branch_wise\\N=5000_burnin=1000_cv=0.99_n=11000_pk=0.1_qa=2_qaa=1_qb=3_qbb=1_r=1.5_s=0.5_ss=0.4_ηa=5_ηb=1_σ=0.5_σ0=0.5.2.counts.csv", delim=",")
# df = df[1:1000, :]
tree = SpeciesTree(".\\PhyloVI\\data\\plants2.nw")
# tree = SpeciesTree(".\\PhyloVI\\data\\plants1a.nw")
#Beluga.set_constantrates!(tree)
Beluga.set_wgdrates!(tree)
data = profile(tree, df)
λ_init = rand(length(tree))
μ_init = rand(length(tree))
η_init = .1
L = 2*length(tree) + 1  # λ, μ and η

# prior = ExpRatesPrior(Exponential(1.), Exponential(1.), Beta(1., 1.), Beta(6., 2.))
prior = prior = Beluga.ExpRatesPrior(
    Exponential(2),
    Exponential(2),
    Beta(1,1),
    Beta(6,2))
Q = MeanFieldGaussian([Normal(0.5, .01) for i in 1:L]) # Initial mean-field distribution
model = ExpRatesModel{Float64}(prior, tree, λ_init, μ_init, η_init)

elbo = ELBO(0)
advi = ADVI(1, 200, 2, 100, VarInfLogger(Vector(), Vector(), Vector(), DataFrame(repeat([Float64[]], 4L + 1))))
_, res = advi(elbo, Q, model, data, .05, 0.1, 0.005)
r = Distributions.params(res)[:μ]
res = model_transform(model)(r)

df_res = advi.logger.df
fin = df_res[end, :]
fin_params = model_transform(model)(collect(fin[1:L]))
rr = CSV.read(".\\PhyloVI\\data\\branch_wise\\1.rates.csv", delim=",")
predictions = measurement.(fin_params.λ, collect(fin[L+1:L+length(tree)]))
cλ = DataFrame((predictions = predictions, truth = rr.λ))
predictions = measurement.(fin_params.μ, collect(fin[L+length(tree)+1:L+2*length(tree)]))
cμ = DataFrame((predictions = predictions, truth = rr.μ))

# Do we do different iterations of the optimization process?
Q = MeanFieldGaussian([r..., ones(L)...])

function plot_optimization()
    labels = ["\\lambda", "\\mu", "\\eta"]
    p = []
    for i in 1:Int((length(Q)-1)/2)
        push!(p, Plots.plot(exp.(df_res[i]), ribbon=df_res[i+L], legend=:none, xticks=:none, yticks=collect(1:10)))
    end
    push!(p, Plots.plot(df_res[end], legend=:none, xticks=:none, yticks=:none))
    Plots.plot(p...)
end
plot_optimization()

# i = 1
# m = exp.(df_res[i])[end]
# s = df_res[i+L][end]
# a = m - s
# b = m + s
# Plots.plot(exp.(df_res[i]), ribbon=df_res[i+L], legend=:none, xticks=:none, yticks=[0, a, m, b])

# p, m = Profile(df, tree)
# prior = ExpRatesPrior(
#     Exponential(1), Exponential(1), Beta(1,1), Beta(6,2))
# chain2 = DLChain(p, prior, tree, m)
# chain2 = mcmc!(chain2, 11000, show_every=100)
#
# λ = (r.λ)
# Plots.histogram(chain2.trace[:λ1], bins=200)
# Plots.vline!([λ[1]])

# plt = Plots.plot(xlim=(1,100))
#
# @gif for i in 1:100
#     d = collect(1:i)
#     Plots.plot(exp.(d))
# end
