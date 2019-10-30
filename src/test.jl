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
Distributions.loglikelihood(d::SimpleModel, x::AbstractVector) = loglikelihood(d.dist, x);
logprior(d::SimpleModel)= 0;
grad_logprior(d::SimpleModel, p::AbstractVector) = [0., 0.]::Vector{Float64};

function grad_loglikelihood(d::SimpleModel, p::AbstractVector{T}, data::AbstractVector{T}) where T<:Real
    # TODO: Maybe implement this...
end

Base.length(d::SimpleModel) = length(d.dist)

function model_transform(d::SimpleModel)
    t = as((μ = asℝ, σ = asℝ₊))
    return t
end
function model_invtransform(d::SimpleModel)
    t = model_transform(d)
    return inverse(t)
end

# Generate some normally distributed data
m_true = -10
s_true = 3
true_posterior = Normal(m_true, s_true)
data = rand(true_posterior, 1000)

# Create a model
model = SimpleModel(Normal(m_true, s_true))

# Create a variational distribution
dists = [Normal(1., 1.1) for i in 1:2]
q = MeanFieldGaussian(dists)

# Test ELBO calculations
elbo = ELBO(100)
elbo(q, model, data)
grad_elbo(q, model, data)
calc_grad_elbo(q, model, data)

# Try ADVI
advi = ADVI(10, 1000)
res = advi(q, model, data, [[1e-2, 1e-3], [1e-3, 1e-3]])
print(res)

# Get the inferred parameters
p = params(res)[:μ]
p = model_invtransform(model)((μ=p[1], σ=p[2]))

using StatsPlots
using Plots
histogram(data, bins=20, normalize=true)
