using Distributions
using StatsBase

#===============================================================================
                            Main file for testing
===============================================================================#

mutable struct SimpleModel <: ContinuousUnivariateDistribution
    dist::Normal
end
function SimpleModel(n::NamedTuple)
    n = Normal(n[:μ][1], n[:σ][1])
    return SimpleModel(n)
end
function Base.rand(d::SimpleModel, N::Int)
    return rand(d.dist, N)
end
Base.rand(d::SimpleModel) = rand(d, 1)
function Distributions.params(d::SimpleModel)
    μ, σ = Distributions.params(d.dist)
    return (μ = μ, σ = σ)
end
function setparams!(d::SimpleModel, μ)
    curr = params(d)
    d.dist = Normal(μ, curr[2])
end
Distributions.loglikelihood(d::SimpleModel, x::AbstractVector) = loglikelihood(d.dist, x);
logprior(d::SimpleModel)= 0;
grad_logprior(d::SimpleModel, p::AbstractVector) = [0., 0.]::Vector{Float64};

function grad_loglikelihood(d::SimpleModel, p::AbstractVector{T}, data::AbstractVector{T}) where T<:Real
    #=δμ = sum((data .- d.dist.μ) / d.dist.σ^2)
    δσ = sum((1.0 ./ d.dist.σ) * (-1.0 .+ (data .- d.dist.μ).^2 ./ d.dist.σ^2))
    return [δμ, δσ]=#

    # Take the gradient of the loglikelihood for the parameters
end

Base.length(d::SimpleModel) = length(d.dist)
function Base.copy(d::SimpleModel)
    p = params(d.dist)
    n = Normal(p[1], p[2])
    s = SimpleModel(n)
    return s
end

function model_transform(d::SimpleModel)
    t = as((μ = asℝ, σ = asℝ₊))
    return t
end
function model_invtransform(d::SimpleModel)
    return inverse(t)
end

# Generate some normally distributed data
m_true = 2.
s_true = 1.
true_posterior = Normal(m_true, s_true)
data = rand(true_posterior, 100)

# Create a model
model = SimpleModel(Normal(m_true, s_true))

# Create a variational distribution
dists = [Normal() for i in 1:2]
q = MeanFieldGaussian(dists)

# Test ELBO calculations
elbo = ELBO(1_000)
elbo(q, model, data)
grad_elbo(q, model, data)
calc_grad_elbo(q, model, data)

# Try ADVI
advi = ADVI(10, 10)
res = advi(q, model, data, 1e-10)
print(res)

using StatsPlots
using Plots
histogram(data, bins=20, normalize=true)
