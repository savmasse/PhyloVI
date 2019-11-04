using Distributions
using StatsBase
using ForwardDiff
using TransformVariables

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
m_true = 10
s_true = 2
true_posterior = Normal(m_true, s_true)
data = rand(true_posterior, 100)

# Create a model
model = SimpleModel(Normal(m_true, s_true))

# Create a variational distribution
dists = [Normal(1., 1.1) for i in 1:2]
q = MeanFieldGaussian(dists)

# Do optimization
elbo = ELBO(100)
advi = ADVI(10, 100, 1, 5, VarInfLogger(Vector(), Vector(), Vector()))
res = advi(elbo, q, model, data, 1., 0.9)
println(res)

# Get the inferred parameters
pars = params(res)[:μ]
pars = model_transform(model)(pars)
println(pars)

# Experiment with plotting the logged values
using StatsPlots
using Plots
using LaTeXStrings
using DataFrames

θ = advi.logger.θ
θ = Array{Float64}(undef, size(θ)[1], length(θ[1]))
∇ = advi.logger.∇
∇ = Array{Float64}(undef, size(∇)[1], length(∇[1]))
# θ[:, 1] = [advi.logger.θ[i][1] for i in 1:size(θ)[1]]
# θ[:, 2] = [advi.logger.θ[i][2] for i in 1:size(θ)[1]]
# θ[:, 3] = [advi.logger.θ[i][3] for i in 1:size(θ)[1]]
# θ[:, 4] = [advi.logger.θ[i][4] for i in 1:size(θ)[1]]
# Plots.plot(advi.logger.objective)

for j in 1:4
    θ[:, j] = [advi.logger.θ[i][j] for i in 1:size(θ)[1]]
    ∇[:, j] = [advi.logger.∇[i][j] for i in 1:size(∇)[1]]
end

# ∇[:, 1] = [advi.logger.∇[i][1] for i in 1:size(∇)[1]]
# ∇[:, 2] = [advi.logger.∇[i][2] for i in 1:size(∇)[1]]
# ∇[:, 3] = [advi.logger.∇[i][3] for i in 1:size(∇)[1]]
# ∇[:, 4] = [advi.logger.∇[i][4] for i in 1:size(∇)[1]]

y = model_transform(model)(rand(2))
d = DataFrame(θ)
# for i in 1:4
#     insert!(d, 4+i, ∇[:, i], Symbol(i))
# end
insert!(d, 5, advi.logger.objective, Symbol(9))

p = []
labels=["\\mu_mean", "\\sigma_mean", "\\mu_stdev", "\\sigma_stdev", "\\delta\\mu_mean", "\\delta\\sigma_mean", "\\delta\\mu_stdev", "\\delta\\sigma_stdev", "ELBO"]
for i in 1:2
    push!(p, Plots.plot(d[i], title=labels[i], legend=:none, ribbon=d[i+2]))
end
push!(p, Plots.plot(d[5], title=labels[9], legend=:none))

Plots.plot(p...)
