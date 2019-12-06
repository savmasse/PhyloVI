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
function SimpleModel(x::AbstractVector{T}) where T<:Real
    n = Normal(x[1], x[2])
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
m_true = 20
s_true = exp(1)
true_posterior = Normal(m_true, s_true)
data = rand(true_posterior, 100)

# Create a model
model = SimpleModel(Normal(m_true, s_true))

# Create a variational distribution
dists = [Normal(1., 1.1) for i in 1:2]
q = MeanFieldGaussian(dists)

# Do optimization
elbo = ELBO(100)
r = repeat([Float64[]], 9)
advi = ADVI(10, 500, 1, 200, VarInfLogger(Vector(), Vector(), Vector(), DataFrame(r)))
res = advi(elbo, q, model, data, .3, 0.1)
println(res)

# Get the inferred parameters
pars = Distributions.params(res)[:μ]
pars = model_transform(model)(pars)
println(pars)

# Experiment with plotting the logged values
using StatsPlots
using Plots
using LaTeXStrings
using DataFrames

df = advi.logger.df
Plots.plot(df[end], legend=:none, title="ELBO")
Plots.plot(df[1], legend=:none)

using Optim

function set_params!(q::MeanFieldGaussian, x)
    μ = x[:μ]
    σ = x[:σ]
    for i in 1:length(q)
        q.dists[i] = Normal(μ[i], σ[i])
    end
end

function set_params!(q::MeanFieldGaussian, x::AbstractVector)
    N = length(q)
    for i in 1:N
        q.dists[i] = Normal(x[i], x[N+i])
    end
end

function o()

    Q = MeanFieldGaussian(2)
    elbo = ELBO(100)
    model = SimpleModel(Normal(m_true, s_true))


    function calc_grad!(G, x)

        # for i in eachindex(x)
        #     if isnan(x[i])
        #         x[i] = eps()
        #     end
        # end

        p = sample_transform(Q)(x)
        p.σ .= max.(p.σ, eps())

        # Update Q with new values
        set_params!(Q, p)

        # Calcalute the gradient
        g = calc_grad_elbo(Q, model, data, 10)
        G[1] = -g[1]
        G[2] = -g[2]
        G[3] = -g[3]
        G[4] = -g[4]

        return G
    end

    function e(x)
        p = sample_transform(q)(x)
        p.σ .= max.(p.σ, eps())
        Qt = MeanFieldGaussian(p)
        y = -elbo(Qt, model, data)
        return y
    end

    # nlprecon = GradientDescent(alphaguess=Optim.LineSearches.InitialStatic(alpha=1e-3,scaled=true), linesearch=Optim.LineSearches.Static())
    # oacc10 = OACCEL(nlprecon=nlprecon, wmax=100)

    init = [1., 1.1, 1.1, 1.1]
    # res = optimize(e, calc_grad!, init, NelderMead())
    res = optimize(e, calc_grad!, init, NelderMead())
    println(res.minimizer)
    println("Distribution: ", Q)
end

o()
