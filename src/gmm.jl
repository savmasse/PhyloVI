
using Distributions

#===============================================================================
                        Define Gaussian Mixture Model
===============================================================================#

mutable struct GaussianMixtureModel{T} <: ContinuousMultivariateDistribution where T<:Real
    μ::Vector{T}
    σ::Vector{T}
    ϕ::Array{T}
end
function GaussianMixtureModel{T}() where T<:Real
    μ = Vector{T}()
    σ = Vector{T}()
    ϕ = Array{T, 1}()
    return GaussianMixtureModel(μ, σ, ϕ)
end
function GaussianMixtureModel{T}(n::NamedTuple) where T<:Real
    μ = n[:μ]
    σ = n[:σ]
    ϕ = n[:ϕ]
    return GaussianMixtureModel(μ, σ, ϕ)
end

function logprior(gmm::GaussianMixtureModel)
    return 0    # Hardcoded for now, so no prior...
end
function Distributions.pdf(gmm::GaussianMixtureModel, x::AbstractVector{T}) where T<:Real
    K = length(gmm.μ)
    n::Vector{Normal} = [Normal(gmm.μ[i], gmm.σ[i]) for i in 1:K]

    r = Vector()
    for i in eachindex(x)
        s = 0.0
        for k in 1:K
            s += gmm.ϕ[i, k] * pdf(n[k], x[i])
        end
        push!(r, s)
    end
    return r
end
Distributions.logpdf(gmm::GaussianMixtureModel, x::AbstractVector{T}) where T<:Real = return log.(pdf(gmm, x))

function Distributions.loglikelihood(gmm::GaussianMixtureModel, x::AbstractVector)
    r = sum(logpdf(gmm, x))
    return r
end

function Distributions.params(gmm::GaussianMixtureModel)
    n = (μ = gmm.μ, σ = gmm.σ, ϕ = gmm.ϕ)
    return n
end

function model_transform(gmm::GaussianMixtureModel)
    K = length(gmm.μ)
    N = size(gmm.ϕ)[1]
    t = as((μ = as(Array, asℝ, K), σ = as(Array, asℝ₊, K), ϕ = as(Array, as_unit_interval, N, K)))
    return t
end
function model_invtransform(gmm::GaussianMixtureModel)
    t = model_transform(gmm)
    return inverse(t)
end

function (elbo::ELBO)(q::M, model::GaussianMixtureModel{T}, data::AbstractVector{T}) where {M<:MeanField, T<:Real}
    r = 0.0
    N = elbo.n_samples
    n = M(length(q.dists))

    for i in 1:N
        η = rand(n)                         # Standardized parameters
        ζ = inv_elliptical(q, η)            # Real space parameters
        transform = model_transform(model)
        θ = transform(ζ)                    # Parameter

        # Normalize parameters
        ϕ = θ[:ϕ]
        N = 100
        for i in 1:N
            ϕ[i,:] ./= sum(ϕ[i,:])
        end
        ζ = model_invtransform(model)(θ)
        m = GaussianMixtureModel{T}(θ)
        r += logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2]
    end

    return r/N + entropy(q)
end

function grad_elbo(q::M, model::GaussianMixtureModel{T}, data::AbstractVector{T}) where {M<:MeanField, T<:Real}

    # Get variational parameters (real space)
    p = [i for i in Iterators.flatten(collect(params(q)))]
    p_real = sample_invtransform(q)(params(q))

    # Create standard normal variational distribution
    n = M(length(q.dists))

    function f(x)
        # Sample the variational distribution
        p = sample_transform(q)(x)          # Variational params in parameter space
        Q = M(p)
        η = rand(n)                         # Sample standard normal
        ζ = inv_elliptical(Q, η)            # Transform to real space
        transform = model_transform(model)
        θ = transform(ζ)                    # Transform to parameter space

        # Normalize parameters
        ϕ = θ[:ϕ]
        N = 100
        for i in 1:N
            ϕ[i,:] ./= sum(ϕ[i,:])
        end
        ζ = model_invtransform(model)(θ)

        m = GaussianMixtureModel{T}(θ)
        return logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2] + entropy(Q)
    end
    g(x) = ForwardDiff.gradient(f, x)

    return g(p_real)
end

#===============================================================================
                        Main program for testing GMM
===============================================================================#

# Set true parameters
K = 2
μ = [-5., 5.]               # True mixture means
σ = [2., 2.]                # True standard deviations
n = 50
N = K*n

# Generate data and set labels
ϕ = zeros(N, K)             # True class labels
data = Vector{Float64}()
for i in 1:K
    ϕ[(i-1)*n+1:i*n, i] = ones(n)
    d = rand(Normal(μ[i], σ[i]), n)
    append!(data, d)
end

# Create a GMM and variational distribution
model = GaussianMixtureModel(μ, σ, ϕ)
#ϕ_init = transpose(rand(Dirichlet(rand(2)), N))
q = MeanFieldGaussian([Normal(2., 1.) for i in 1:(2*N+2*K)])

# Create ELBO and ADVI objects
elbo = ELBO(50)
advi = ADVI(1, 100)

res = advi(q, model, data, [rand(204), rand(204)])
p = params(res)[:μ]

θ = model_transform(model)(p)
println(θ[:ϕ])
ϕ = θ[:ϕ]
for i in 1:N
    ϕ[i,:] ./= sum(ϕ[i,:])
end
println(ϕ)
