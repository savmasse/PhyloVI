
using Distributed
using LinearAlgebra
abstract type VariationalCostFunction end

#===============================================================================
                    ELBO type definition and functions
===============================================================================#
mutable struct ELBO <: VariationalCostFunction
    n_samples::Int  # Number of Monte carlo samples for ELBO computation
end
ELBO() = ELBO(1)

function (elbo::ELBO)(q::Q, model::M, data::AbstractArray, batch_factor::Float64=1.0) where {Q<:MeanField, M<:PhyloModel}

    # Take N real-space samples from the variational distribution and transform
    # to parameter space.
    N = elbo.n_samples
    ζ = rand(q, N)
    T = model_transform(model)
    θ = [T(ζ[i]) for i in 1:N]

    # Take ELBO N times and take mean
    r = 0.0
    for i in 1:N
        m = model(θ[i])
        r += logprior(m) + loglikelihood(m, data) * batch_factor
    end
    return r/N + entropy(q)
end

# TODO: This function should be made more generic so it works for any T<:MeanField
function grad_elbo(q::MeanFieldGaussian, model::M, data::AbstractArray, batch_factor::Float64=1.0) where {M<:PhyloModel}

    # Check inputs
    if batch_factor < 1.0
        throw("Batch factor should be in domain [1.0; ∞[")
    end

    # Sample unit variational function and transform
    p = sample_invtransform(q, Distributions.params(q))
    n = MeanFieldGaussian(length(q))
    η = rand(n)
    ζ = inv_elliptical(q, η)
    T = model_transform(model)
    θ = T(ζ)
    m = model(θ)

    # Put the components together
    # γ = (grad_logprior(m) .+ grad_loglikelihood(m, data) .* batch_factor) .* grad_invtransform(m, ζ) .+ grad_logdetjac(m, ζ)
    γ = (grad_logprior(m) .+ grad_loglikelihood(m, data) .* batch_factor) .+ grad_logdetjac(m, ζ)
    μ = γ
    ω = γ .* (η .* p[2])
    return Float64[μ..., ω...] .+ grad_entropy(n)
end

function calc_grad_elbo(q::M, model::D, data, N::Int=10, batch_factor::Float64=1.0) where {M<:MeanField, D<:PhyloModel}

    n = dimension(sample_transform(q))
    r = zeros(Float64, n)

    for i in 1:N
        r .+= grad_elbo(q, model, data, batch_factor)
    end
    r ./= N

    return r
end
