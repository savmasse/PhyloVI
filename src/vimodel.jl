
using ForwardDiff

#===============================================================================
        Variational model functioning as a generic interface to perform
        VI on any model, not necessarily only phylogenetic models.

        This interface defines methods for the calculation of the logprior
        and loglikelihood and their gradients.
        Also contains model parameter transformations to real space and
        back to parameter space.
===============================================================================#

# TODO: Implement the main functionality here instead of in PhyloModel, not all
# model are PhyloModels but they do need transforms and likelihoods/priors...
# TODO: Add  "model(x)" function that allows easy setting of parameters
# TODO: Add "asvector(model)" function to get the parameters

abstract type VIModel end

function logprior(model::T, data::AbstractArray) where {T<:VIModel}
    throw("VI model logprior method not implemented.")
end

function loglikelihood(model::T, data::AbstractArray) where {T<:VIModel}
    throw("VI model loglikelihood method not implemented.")
end

function grad_logprior(model::T, data::AbstractArray) where {T<:VIModel}
    return ForwardDiff.gradient(x -> logprior(model, x), data)
end

function grad_loglikelihood(model::T, data::AbstractArray) where {T<:VIModel}
    return ForwardDiff.gradient(x -> loglikelihood(model, x), data)
end

# New transform interface functions
function model_transform(model::T) where T<:VIModel
    # Should return a transform function
    throw("VI model parameter transform method not implemented.")
end
model_transform(model::T, x::AbstractArray{V}) where {T<:VIModel, V<:Real} =
    model_transform(model)(x)
