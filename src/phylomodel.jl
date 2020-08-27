
using ForwardDiff
using Distributions

#===============================================================================
        Generic Phylogenetic model that implements the calculation of
        logprior, loglikelihood, their gradients, and the transformations
        to real space and back to parameter space.

        A PhyloModel should contain a prior distribution, a Beluga/Whale
        model and a species tree.

        Defines the following functions:

        - logprior: log of prior of the current model parameters
        - loglikelihood: log of likelihood of current model parameters and data
        - grad_logprior: gradient of logprior
        - grad_loglikelihood: gradient of loglikelihood
        - model_transform: transform from real space to parameter space
        - model_invtransform: transform from parameter space to real space
        - params: get current parameters of the model

        A PhyloModel should contain a "prior" and a "model". These are
        assumed to exist in the interface implementations.
===============================================================================#

# TODO:: Do we really need this step in-between VIModel and higher models??
# What advantage does this struct have over VIModel?

abstract type PhyloModel <: VIModel end
const PhyloData{T} = AbstractVector{T} # TODO:: Get rid of this data type idea...

# function logprior(model::T) where T<:PhyloModel
#     throw("Logprior method not yet implemented for this phylogenetic model.")
# end
#
# function loglikelihood(model::T, data::AbstractArray) where {T<:PhyloModel}
#     throw("Loglikelihood method not yet implemented for this phylogenetic model.")
# end

function grad_logprior(model::T) where T<:PhyloModel

    # Get current model parameters
    θ = params(model)
    ζ = model_invtransform(model)(θ)

    # Define helper function for gradient
    lp(x) = begin
        θ = collect(Iterators.flatten(model_transform(model, x)))
        return logprior(model(θ))
    end

    return ForwardDiff.gradient(lp, ζ)
end

function grad_loglikelihood(model::T, data::AbstractArray) where {T<:PhyloModel}

    # Get current model parameters
    θ = params(model)
    ζ = model_invtransform(model)(θ)

    # Define helper function for gradient
    ll(x) = begin
        θ = collect(Iterators.flatten(model_transform(model, x)))
        return loglikelihood(model(θ), data)
    end

    return ForwardDiff.gradient(ll, ζ)
end

function model_transform(model::T) where T<:PhyloModel
    throw("Model transform not implemented for this type of model.")
end

function model_invtransform(model::T) where T<:PhyloModel
    throw("Inverse model transform not implemented for this type of phylogenetic model")
end

function setparams!(model::T, θ) where T<:PhyloModel
    throw("Parameter update method not implemented for this type of phylogenetic model.")
end

function params(model::T) where T<:PhyloModel
    throw("params method not yet implemented.")
end
