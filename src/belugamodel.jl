
using TransformVariables
using Beluga
using DistributedArrays

struct BelugaModel <: PhyloModel
    prior
    model
end

function (model::BelugaModel)(θ::AbstractVector{T}) where T<:Real
    return BelugaModel(model.prior, model.model(θ))
end
function (model::BelugaModel)(θ::NamedTuple)
    p = [i for i in Iterators.flatten(θ)]
    return BelugaModel(model.prior, model.model(p))
end

function params(model::BelugaModel)
    p = asvector(model.model)
    L = length(model.model)
    return (λ = p[1:L],  μ = p[L+1:2L], η = p[end])
end

function setparams!(model::BelugaModel, θ::NamedTuple)
    λ = θ[1]
    μ = θ[2]
    η = θ[3]

    x = [λ..., μ..., η]
    # L = length(model.model)
    # update!(model.model[1], (μ=μ[1], λ=λ[1], η=η))
    # for i in 2:L
    #     update!(model.model[i], (μ=μ[i], λ=λ[i]))
    # end
end
function setparams!(model::BelugaModel, θ::AbstractVector{T}) where T<:Real
    L = length(model.model)
    λ = θ[1:L]
    μ = θ[L+1:2L]
    η = θ[end]

    x = [λ..., μ..., η]
    model = model.model(x)
    # # update!(model.model[1], (μ=μ[1], λ=λ[1], η=η))
    # # for i in 2:L
    # #     update!(model.model[i], (μ=μ[i], λ=λ[i]))
    # # end
    # x = model.model[1].x.θ
    # y = Dict{Symbol, Real}(:μ=>x[:μ], :λ=>x[:λ], :t=>x[:t], :η=>η)
    # model.model[1].x.θ = y
    # for i in 1:L
    #     x = model.model[i].x.θ
    #     y = Dict{Symbol, Real}(:μ=>μ[i], :λ=>λ[i], :t=>x[:t], :η=>x[:η])
    #     model.model[i].x.θ = y
    # end
end

function logprior(model::BelugaModel)
    return logpdf(model.prior, model.model)
end

function loglikelihood(model::BelugaModel, data::AbstractVector{T}) where T
    return logpdf!(model.model, data)
end

function grad_logprior(model::BelugaModel)
    return Beluga.gradient(model.prior, model.model)
end

function grad_loglikelihood(model::BelugaModel, data::AbstractVector{T}) where T
    return Beluga.gradient(model.model, data)
end

function grad_invtransform(model::BelugaModel, ζ)
    L = length(model.model)
    λ = exp.(ζ[1:L])
    μ = exp.(ζ[L+1:2L])
    η = ζ[end] > 0.0 ? exp(-ζ[end]) : exp(ζ[end])
    return [λ..., μ..., η]
end

function grad_logdetjac(model, ζ)
    T = model_transform(model)
    helper(x) = begin
        return TransformVariables.transform_and_logjac(T, x)[2]
    end
    return ForwardDiff.gradient(helper, ζ)
end

function model_transform(model::BelugaModel)
    N = length(model.model)
    t = as((λ = as(Array, as_positive_real, N),
            μ = as(Array, as_positive_real, N),
            η = as_unit_interval))
    return t
end

function model_invtransform(model::BelugaModel)
    t = inverse(model_transform(model))
    return t
end
