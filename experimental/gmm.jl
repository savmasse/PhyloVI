using Beluga
using PhyloVI
using DataFrames
using BenchmarkTools

include("../src/vimodel.jl")
include("../src/phylomodel.jl")
include("../src/meanfield.jl")
include("../src/elbo.jl")
include("../src/logger.jl")
include("../src/advi.jl")
include("../src/wgdbelugamodel.jl")

#===============================================================================
                            Gaussian Mixture Model
===============================================================================#

struct GaussianMixtureModel{T} <: PhyloModel where T<:Real
    K::Int
    μ::Array{T}
    σ::Array{T}
    π::Array{T}
end
GaussianMixtureModel(K::Int) = GaussianMixtureModel{Float64}(K, zeros(K), ones(K), repeat([1/K], K))
Base.length(model::GaussianMixtureModel) = 3model.K

function (model::GaussianMixtureModel)(θ::Array{T}) where T<:Real
    if length(θ) != 3*model.K
        throw("Invalid length of parameter vector for the given model.")
    end

    μ = θ[1:model.K]
    σ = θ[model.K+1:2*model.K]
    π = θ[2*model.K+1:end]

    return GaussianMixtureModel(model.K, μ, σ, π)
end

function (model::GaussianMixtureModel)(nt::NamedTuple)
    μ, σ, π = nt
    return GaussianMixtureModel{Float64}(length(μ), μ, σ, π)
end

function PhyloVI.asvector(model::GaussianMixtureModel)
    return [model.μ..., model.σ..., model.π...]
end

function params(model::GaussianMixtureModel)
    θ = asvector(model)
    K = model.K
    return (μ = θ[1:K], σ = θ[K+1:2K], π = θ[2K+1:end])
end

function logprior(model::GaussianMixtureModel)
    # d = Dirichlet(repeat([1/model.K], model.K))
    lp = sum(logpdf.(Normal(0, 2), model.μ))
    lp += sum(logpdf.(LogNormal(-1, .5), model.σ))
    lp += sum(repeat([log(1/model.K)], model.K))
    return lp
end

function loglikelihood(model::GaussianMixtureModel, data::AbstractArray{T}) where T<:Real
    ll = zero(T)
    for i in 1:length(data)
        temp = zero(T)
        for k in 1:model.K
            temp += (model.π[k] * exp(logpdf(Normal(model.μ[k], model.σ[k]), data[i])))
        end
        ll += log(temp)
    end
    return ll
end

# # TODO: This is standard use that should go into standard VIModel struct...
# # Should not have to overload this...
# function grad_logprior(model::GaussianMixtureModel)
#
#     lp(x) = begin
#         m = model(x)
#         return logprior(m)
#     end
#
#     θ = asvector(model)
#     ForwardDiff.gradient(lp, θ)
# end
#
# # TODO: This is standard use that should go into standard VIModel struct...
# # Should not have to overload this...
# function grad_loglikelihood(model::GaussianMixtureModel, data::AbstractArray{T}) where T<:Real
#
#     ll(x) = begin
#         m = model(x)
#         return loglikelihood(m, data)
#     end
#
#     θ = asvector(model)
#     ForwardDiff.gradient(ll, θ)
# end

# TODO: This is standard use that should go into standard VIModel struct...
# Should not have to overload this...
function grad_logdetjac(model::GaussianMixtureModel, ζ)
    T = model_transform(model)
    helper(x) = begin
        return TransformVariables.transform_and_logjac(T, x)[2]
    end
    return ForwardDiff.gradient(helper, ζ)
end

function model_transform(model::GaussianMixtureModel)
    t = as((μ = as(Array, as_real, model.K),
            σ = as(Array, as_positive_real, model.K),
            # π = as(Array, as_unit_interval, model.K)))
            π = UnitSimplex(model.K)))
    return t
end

function model_invtransform(model::GaussianMixtureModel)
    t = inverse(model_transform(model))
    return t
end

function grad_invtransform(model::GaussianMixtureModel, ζ)
    K = model.K
    μ = ones(K)
    q = exp.(ζ[K+1:2K])
    π = [(ζ[i] > 0.0 ? exp(-ζ[i]) : exp(ζ[i])) for i in 2K+1:3K]
    return [μ..., σ..., π...]
end


################################################################################

# Create some artificial data
function Base.rand(model::GaussianMixtureModel)
    c = Categorical(model.π)
    i = rand(c)
    return rand(Normal(model.μ[i], model.σ[i]))
end

function Base.rand(model::GaussianMixtureModel, N::Int)
    samples = Float64[]
    n = [Normal(model.μ[i], model.σ[i]) for i in 1:model.K]
    for c in rand(Categorical(model.π), N)
        push!(samples, rand(n[c]))
    end
    return samples
end
