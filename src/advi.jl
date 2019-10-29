
abstract type VariationalAlgorithm end

#===============================================================================
                    ADVI type definition and functions
===============================================================================#

mutable struct ADVI
    n_samples::Int  # Amount of Monte Carlo samples in gradient computation
    max_iter::Int
end

function (advi::ADVI)(q::M, model::D, data::AbstractVector{T}, δ=1e-4) where {M<:MeanField, D<:Distribution, T<:Real}

    # Setup/preprocessing
    μ, σ = params(q)
    N = Int(dimension(sample_transform(q))/2)
    Q = M((μ=μ, σ=σ))

    # Enter convergence loop
    for i in 1:advi.max_iter

        # Show elbo (TODO: Fix this ugly hack!!)
        elbo = ELBO(100)
        e = elbo(q, model, data)
        println("ELBO: ", e)

        # Calculate the gradient
        c = calc_grad_elbo(q, model, data, advi.n_samples)
        μ₀ = c[1:N]
        σ₀ = c[N+1:end]

        # Update the parameters (this is real space)
        μ .+= δ*μ₀
        σ .+= δ*σ₀

        # Transform back to parameter space
        ζ = [μ..., σ...]
        μ_new, σ_new = sample_transform(q)(ζ)

        # Update q
        q = M((μ=μ_new, σ=σ_new))
        println(q)
    end

    return q
end
