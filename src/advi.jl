
abstract type VariationalAlgorithm end

#===============================================================================
                    ADVI type definition and functions
===============================================================================#

mutable struct ADVI
    n_samples::Int  # Amount of Monte Carlo samples in gradient computation
    max_iter::Int
end

function (advi::ADVI)(q::M, model::D, data::AbstractVector{T}, δ=[1e-3, 1e-3]) where {M<:MeanField, D<:Distribution, T<:Real}

    # Setup/preprocessing
    # Transform parameters to real space
    θ = sample_invtransform(q)(params(q))
    N = Int(dimension(sample_transform(q))/2)
    μ = θ[1:N]
    ω = θ[N+1:end]

    prev = 0
    elbo = ELBO(100)
    counter = 0
    patience = 0
    q_best = q
    Q = q

    # Enter convergence loop
    for i in 1:advi.max_iter

        if patience >= 3
            break
        end

        counter += 1

        curr = elbo(Q, model, data)
        println("ELBO: ", curr)

        # Calculate the gradient
        c = calc_grad_elbo(Q, model, data, advi.n_samples)
        μ₀ = c[1:N]
        ω₀ = c[N+1:end]

        # Update the parameters (this is still real space)
        μ .+= δ[1].*μ₀
        ω .+= δ[2].*ω₀

        # Transform back to parameter space
        ζ = [μ..., ω...]
        μ_new, σ_new = sample_transform(q)(ζ)

        if curr < prev
            patience += 1
        else
            patience = 0
            q_best = M((μ=μ_new, σ=σ_new))
        end
        prev = curr

        Q = M((μ=μ_new, σ=σ_new))
    end

    println("Finished ADVI after ", counter, " iterations.")
    println("Final ELBO: ", elbo(q_best, model, data))
    return q_best
end
