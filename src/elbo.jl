
abstract type VariationalCostFunction end

#===============================================================================
                    ELBO type definition and functions
===============================================================================#
mutable struct ELBO <: VariationalCostFunction
    n_samples::Int  # Number of Monte carlo samples for ELBO computation
end
ELBO() = ELBO(10)

# Calculate the ELBO of a certain model and variational distribution
function (elbo::ELBO)(q::M, model::D, data::AbstractVector{T}) where {M<:MeanField, D<:Distribution, T<:Real}
    r = 0.0
    N = elbo.n_samples

    for i in 1:N
        ζ = rand(q)
        θ = model_transform(model)(ζ)
        m = D(θ)
        r += logprior(m) + loglikelihood(m, data)
    end
    return r/N - entropy(q)
end

function grad_elbo(q::M, model::D, data::AbstractVector{T}) where {M<:MeanField, D<:Distribution, T<:Real}

    # Get variational parameters
    p = [i for i in Iterators.flatten(collect(params(q)))]

    function f(x)
        # Sample the variational distribution
        Q = M(x)
        ζ = rand(Q)
        # Transform sampled parameters to parameter space
        transform = model_transform(model)
        θ = transform(ζ)
        m = D(θ)
        # Calculate the ELBO
        return logprior(m) + loglikelihood(m, data) +  TransformVariables.transform_and_logjac(transform, ζ)[2] - entropy(q)
    end
    g(x) = ForwardDiff.gradient(f, x)

    return g(p)
end

function calc_grad_elbo(q::M, model::D, data::AbstractVector{T}, N::Int=1) where {M<:MeanField, D<:Distribution, T<:Real}

    n = dimension(sample_transform(q))
    r = zeros(T, n)

    for i in 1:N
        r .+= grad_elbo(q, model, data)
    end
    r ./= N

    return r
end
