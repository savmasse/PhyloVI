
abstract type VariationalAlgorithm end

#===============================================================================
                    ADVI type definition and functions
===============================================================================#

mutable struct ADVI
    n_samples::Int  # Amount of Monte Carlo samples in gradient computation
    max_iter::Int
end

function (advi::ADVI)(elbo::ELBO, q::M, model::D, data::AbstractVector{T}, η::Float64=1.0, α::Float64=.1,
    verbose=0, patience=5) where {M<:MeanField, D<:Distribution, T<:Real}

    # Transform parameters to real space
    ζ = sample_invtransform(q)(params(q))
    N = Int(dimension(sample_transform(q)))

    # Setup some parameters
    prev = -Inf
    best_elbo = -Inf
    counter = 0
    p = 0
    q_best = q
    Q = q
    s = zeros(N)

    # Enter convergence loop
    for i in 1:advi.max_iter

        # Check if should exit loop
        if p >= patience
            break
        end
        counter += 1

        # Calculate the ELBO
        curr = elbo(Q, model, data)
        if verbose > 1; println("Iteration ", i, ": ", curr); end

        # Calculate the gradient and update parameters
        g = calc_grad_elbo(Q, model, data, advi.n_samples)
        δ = calc_step(i, η, α, g, s)
        ζ .+= δ .* g
        μ_new, σ_new = sample_transform(q)(ζ)

        # Check if we should stop
        if curr < best_elbo
            p += 1
        else
            p = 0
            q_best = M((μ=μ_new, σ=σ_new))
            best_elbo = curr
        end
        prev = curr

        # Update the parameters of the variational distribution
        Q = M((μ=μ_new, σ=σ_new))
    end

    if verbose > 0
        println("Finished ADVI after ", counter, " iterations.")
        println("Final ELBO: ", best_elbo)
    end
    return q_best
end

function calc_step(i::Int, η::Float64, α::Float64, g::AbstractVector{T}, s::AbstractVector{T}) where T<:Real
    ϵ = eps()
    s = (i == 1) ? g.^2 : α^2 .* g.^2 .+ (1-α) .* s
    δ_new = η .* i^(-1/2 + ϵ) ./ (1.0 .+ s.^(1/2))
    return δ_new
end
