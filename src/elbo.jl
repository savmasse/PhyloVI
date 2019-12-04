
abstract type VariationalCostFunction end

#===============================================================================
                    ELBO type definition and functions
===============================================================================#
mutable struct ELBO <: VariationalCostFunction
    n_samples::Int  # Number of Monte carlo samples for ELBO computation
end
ELBO() = ELBO(10)

# Calculate the ELBO of a certain model and variational distribution
function (elbo::ELBO)(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}

    N = elbo.n_samples
    if N == 0; return 0.0; end
    r = 0.0
    n = M(length(q.dists))

    for i in 1:N
        η = rand(n)                         # Standardized parameters
        ζ = inv_elliptical(q, η)            # Real space parameters
        transform = model_transform(model)
        θ = transform(ζ)                    # Parameter space
        m = D(θ)
        r += logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2]
    end

    return r/N - entropy(q)
end

function grad_elbo(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}

    # Get variational parameters (real space)
    p_real = sample_invtransform(q)(Distributions.params(q))

    # Create standard normal variational distribution
    n = M(length(q.dists))

    function f(x)
        # Sample the variational distribution
        p = sample_transform(q)(x)          # Variational params in parameter space
        Q = M(p)
        η = rand(n)                         # Sample standard normal
        ζ = inv_elliptical(Q, η)            # Transform to real space
        transform = model_transform(model)
        θ = transform(ζ)                    # Transform to parameter space
        m = D(θ)
        return logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2] - entropy(Q)
    end
    g(x) = ForwardDiff.gradient(f, x)

    return g(p_real)
end

function grad_elbo(q::M, model::D, data::AbstractArray{T}, batch_factor::Float64=1.0) where {M<:MeanField, D<:Distribution, T<:Real}

    # Get variational parameters (real space)
    p_real = sample_invtransform(q)(Distributions.params(q))

    # Create standard normal variational distribution
    N = length(q.dists)
    n = M(N)
    p = sample_transform(q)(p_real)     # Variational params in parameter space
    Q = M(p)
    η = rand(n)                         # Sample standard normal
    ζ = inv_elliptical(Q, η)            # Transform to real space
    transform = model_transform(model)
    θ = transform(ζ)

    # Create vector of theta
    θ = [i for i in Iterators.flatten(θ)]

    function grad_logprior(θ)
        f(x) = begin
            m = D(x)
            return logprior(m)
        end
        return ForwardDiff.gradient(f, θ)
    end
    function grad_loglikelihood(θ)
        f(x) = begin
            m = D(x)
            return loglikelihood(m, data)
        end
        return ForwardDiff.gradient(f, θ)
    end
    function grad_logdetjac(ζ)
        f(x) = begin
            return TransformVariables.transform_and_logjac(transform, x)[2]
        end
        return ForwardDiff.gradient(f, ζ)
    end
    function grad_invtransform(ζ)
        return ones(N)
    end
    function grad_entropy()
        return [zeros(N)..., ones(N)...]
    end

    x = ((grad_logprior(θ) .+ grad_loglikelihood(θ) * batch_factor) .* grad_invtransform(ζ) .+ grad_logdetjac(ζ))
    μ = x
    ω = x .* (η .* p[2])
    return Float64[μ..., ω...] .+ grad_entropy()
end

function calc_grad_elbo(q::M, model::D, data::AbstractArray{T}, N::Int=10, batch_factor::Float64=1.0) where {M<:MeanField, D<:Distribution, T<:Real}

    n = dimension(sample_transform(q))
    r = zeros(T, n)

    for i in 1:N
        r .+= grad_elbo(q, model, data, batch_factor)
    end
    r ./= N

    return r
end
