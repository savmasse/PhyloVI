using Plots
using StatsPlots
using Distributions
using ForwardDiff

#===============================================================================
                        Define Gaussian Mixture Model
===============================================================================#

mutable struct GaussianMixtureModel{T} <: ContinuousMultivariateDistribution where T<:Real
    μ::Vector{T}
    σ::Vector{T}
    ϕ::Vector{T}
end
function GaussianMixtureModel{T}(n::NamedTuple) where T<:Real
    μ = n[:μ]
    σ = n[:σ]
    ϕ = n[:ϕ]
    ϕ ./= sum(ϕ)
    return GaussianMixtureModel(μ, σ, ϕ)
end
function GaussianMixtureModel{T}(x::AbstractVector{V}) where {T<:Real, V<:Real}
    K = Int(length(x)/3)
    μ = x[1:K]
    σ = x[K+1:2K]
    ϕ = x[2K+1:end]
    ϕ ./= sum(ϕ)
    return GaussianMixtureModel(μ, σ, ϕ)
end
Base.length(gmm::GaussianMixtureModel) = length(gmm.μ)

function logprior(gmm::GaussianMixtureModel)
    K = length(gmm)
    dμ = Normal(0., 10.)
    dσ = Exponential(0.5)
    dϕ = Beta(1/K, 6)
    lp = sum(logpdf.(dμ, gmm.μ))
    lp += sum(logpdf.(dσ, gmm.σ))
    lp += sum(logpdf.(dϕ, gmm.ϕ))
    return lp
end
function Distributions.pdf(gmm::GaussianMixtureModel, x::T) where T<:Real
    K = length(gmm)
    r = 0.0
    dists = [Normal(gmm.μ[i], gmm.σ[i]) for i in 1:K]
    for i in 1:K
        r += gmm.ϕ[i] * pdf(dists[i], x)
    end
    return r
end
Distributions.logpdf(gmm::GaussianMixtureModel, x::T) where T<:Real = return log(pdf(gmm, x))
function Distributions.loglikelihood(gmm::GaussianMixtureModel, x::AbstractVector{T}) where T<:Real
    r = 0.0
    for i in eachindex(x)
        r += logpdf(gmm, x[i])
    end
    return r
end
function Distributions.params(gmm::GaussianMixtureModel)
    n = (μ = gmm.μ, σ = gmm.σ, ϕ = gmm.ϕ)
    return n
end
function model_transform(gmm::GaussianMixtureModel)
    K = length(gmm.μ)
    N = size(gmm.ϕ)[1]
    t = as((μ = as(Array, asℝ, K), σ = as(Array, asℝ₊, K), ϕ = as(Array, as_unit_interval, K)))
    return t
end
function model_invtransform(gmm::GaussianMixtureModel)
    t = model_transform(gmm)
    return inverse(t)
end

function grad_elbo(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}

    # Get variational parameters (real space)
    p_real = sample_invtransform(q)(Distributions.params(q))

    # Create standard normal variational distribution
    N = length(q)
    K = length(model)
    n = M(N)
    p = sample_transform(q)(p_real)     # Variational params in parameter space
    Q = M(p)
    η = rand(n)                         # Sample standard normal
    ζ = inv_elliptical(Q, η)            # Transform to real space
    transform = model_transform(model)
    θ = transform(ζ)

    # Create vector of theta
    θ = Float64[i for i in Iterators.flatten(θ)]

    function grad_logprior(θ)
        f(x) = begin
            m = D(x)
            return logprior(m)
        end
        # println(θ)
        return ForwardDiff.gradient(f, θ)
    end
    function grad_loglikelihood(θ)
        f(x) = begin
            m = D(x)
            return loglikelihood(m, data)
        end
        # println(θ)
        return ForwardDiff.gradient(f, θ)
    end
    function grad_logdetjac(ζ)
        f(x) = begin
            return TransformVariables.transform_and_logjac(transform, x)[2]
        end
        # println(ζ)
        return ForwardDiff.gradient(f, ζ)
    end
    function grad_invtransform(ζ)
        μ = ones(K)
        σ = exp.(ζ[K+1:2K])
        ϕ = zeros(K)
        for i in 1:K
            ϕ[i] = (ζ[i] > 0) ? exp(-ζ[i]) : exp(ζ[i])
        end
        # println(ζ)
        return [μ..., σ..., ϕ...]
    end
    function grad_entropy()
        return [zeros(N)..., ones(N)...]
    end

    #println(grad_logprior(θ), "; ", grad_loglikelihood(θ), "; ", grad_invtransform(ζ), "; ", grad_logdetjac(ζ))

    x = ((grad_logprior(θ) .+ grad_loglikelihood(θ)) .* grad_invtransform(ζ) .+ grad_logdetjac(ζ))
    δμ = x
    δω = x .* (η .* p[2])
    return [δμ..., δω...] .+ grad_entropy()
end

#===============================================================================
                        Main program for GMM testing
===============================================================================#

# Set true parameters
K = 3
μ = [-10., 10., -5., 5., -1., 1.]                 # True mixture means
σ = [1., 1., .5, .5, .1, .1]                    # True standard deviations
ϕ = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
N = 6000
# Generate data and set labels
ϕ_true = zeros(N, K)            # True class labels
data = Vector{Float64}()
for i in 1:K
    d = rand(Normal(μ[i], σ[i]), Int(ϕ[i]*N))
    append!(data, d)
end

μ_init = repeat([0.0], K)
σ_init = repeat([1.0], K)
ϕ_init = repeat([1.0/K], K)
model = GaussianMixtureModel{Float64}(μ_init, σ_init, ϕ_init)
q = MeanFieldGaussian([Normal(0.1, 1.) for i in 1:3K])

elbo = ELBO(100)
advi = ADVI(1, 1000, 2, 1000, VarInfLogger([Vector() for i in 1:3]..., DataFrame(repeat([Float64[]], 12K+1))))
res = advi(elbo, q, model, data, 0.1, 0.1)
res = Distributions.params(res)[:μ]
println(res)
c = model_transform(model)(res)
println(c)

# Convert \phi to correct scale
y = GaussianMixtureModel{Float64}(c)

df = advi.logger.df
p = []
for i in 1:K
    push!(p, Plots.plot(df[i], ribbon=df[i+3K], legend=:none))
end
push!(p, Plots.plot(df[end], legend=:none))
Plots.plot(p...)
