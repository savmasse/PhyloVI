using TransformVariables
using Beluga

struct WGDBelugaModel <: PhyloModel
    prior
    model
end
Base.length(model::WGDBelugaModel) = length(model.model)

function (model::WGDBelugaModel)(θ::AbstractVector{T}) where T<:Real
    return WGDBelugaModel(model.prior, model.model(θ))
end
function (model::WGDBelugaModel)(θ::NamedTuple)
    p = [i for i in Iterators.flatten(θ)]
    return WGDBelugaModel(model.prior, model.model(p))
end

function params(model::WGDBelugaModel)
    p = asvector(model.model)
    Q = getwgdcount(model)
    L = length(model) - 2Q
    if Q > 0
        return (λ = p[1:L],  μ = p[L+1:2L], q = p[2L+1:end-1], η = p[end])
    else
        return (λ = p[1:L],  μ = p[L+1:2L], η = p[end])
    end
end

function setparams!(model::WGDBelugaModel, θ::NamedTuple)
    λ = θ[1]
    μ = θ[2]
    q = θ[3]
    η = θ[4]
    x = [λ..., μ..., q..., η]
end
function setparams!(model::WGDBelugaModel, θ::AbstractVector{T}) where T<:Real
    Q = getwgdcount(model)
    L = length(model) - 2Q
    λ = θ[1:L]
    μ = θ[L+1:2L]
    q = θ[2L+1:end-1]
    η = θ[end]
    x = [λ..., μ..., q..., η]
    model = model.model(x)
end

function getwgdcount(model::WGDBelugaModel)
    return length(Beluga.getwgds(model.model))
end

function logprior(model::WGDBelugaModel)
    return logpdf(model.prior, model.model)
end

function loglikelihood(model::WGDBelugaModel, data::AbstractVector{T}) where T
    return logpdf!(model.model, data)
end

function grad_logprior(model::WGDBelugaModel)
    return Beluga.gradient(model.prior, model.model)
end

function grad_loglikelihood(model::WGDBelugaModel, data::AbstractVector{T}) where T
    return Beluga.gradient(model.model, data)
end

# TODO: get rid of this? Is deprecated now...
# This sort of manual definition doesn't work for more complex transforms like Cholesky and simplex...
function grad_invtransform(model::WGDBelugaModel, ζ)
    Q = getwgdcount(model)
    L = length(model) - 2Q
    λ = exp.(ζ[1:L])
    μ = exp.(ζ[L+1:2L])
    q = [(ζ[i] > 0.0 ? exp(-ζ[i]) : exp(ζ[i])) for i in 2L+1:length(ζ)-1]
    η = ζ[end] > 0.0 ? exp(-ζ[end]) : exp(ζ[end])
    return [λ..., μ..., q..., η]
end

function grad_logdetjac(model, ζ)
    T = model_transform(model)
    helper(x) = begin
        return TransformVariables.transform_and_logjac(T, x)[2]
    end
    return ForwardDiff.gradient(helper, ζ)
end

function model_transform(model::WGDBelugaModel)
    Q = length(Beluga.getwgds(model.model))
    L = length(model) - 2Q

    t = nothing
    if Q > 0
        t = as((λ = as(Array, as_positive_real, L),
                μ = as(Array, as_positive_real, L),
                q = as(Array, as_unit_interval, Q),
                η = as_unit_interval))
    else
        t = as((λ = as(Array, as_positive_real, L),
                μ = as(Array, as_positive_real, L),
                η = as_unit_interval))
    end
    return t
end

function model_invtransform(model::WGDBelugaModel)
    t = inverse(model_transform(model))
    return t
end
