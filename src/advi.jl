
abstract type VariationalAlgorithm end

#===============================================================================
                            ADVI optimization logger
===============================================================================#

mutable struct VarInfLogger
    θ::Vector           # Parameters
    ∇::Vector           # Gradients in real space
    objective::Vector   # Variational objective
end

function update_logger!(logger::VarInfLogger, θ, ∇, objective)
    # Add new info to the logger
    push!(logger.θ, θ)
    push!(logger.∇, ∇)
    push!(logger.objective, objective)
end

function display(logger::VarInfLogger)
    # Show logger info in text in nice overview: probably a dataframe
end

function plot(logger::VarInfLogger)
    # Plot the parameters of the logger
end

#===============================================================================
                    ADVI type definition and functions
===============================================================================#

mutable struct ADVI
    n_samples::Int          # Amount of Monte Carlo samples in gradient computation
    max_iter::Int           # Maximum amount of iterations in optimization
    verbose::Int            # Degree of output text during optimization
    patience::Int           # Tolerance for non-improving iterations
    logger::VarInfLogger    # Logger for optimization process
end

function (advi::ADVI)(elbo::ELBO, q::M, model::D, data::AbstractArray{T}, η::Float64=1.0, α::Float64=.1) where {M<:MeanField, D<:Distribution, T<:Real}

    # Transform parameters to real space
    ζ = sample_invtransform(q)(params(q))
    N = Int(dimension(sample_transform(q)))

    # Setup some parameters
    prev_elbo = -Inf
    best_elbo = -Inf
    counter = 0
    p = 0
    q_best = q
    Q = q
    s = zeros(N)

    # Enter convergence loop
    for i in 1:advi.max_iter

        # Check if should exit loop
        if p >= advi.patience
            break
        end
        counter += 1

        # Calculate the ELBO
        curr_elbo = elbo(Q, model, data)
        if advi.verbose > 1; println("Iteration ", i, ": ", curr_elbo); end

        # Calculate the gradient and update parameters
        g = calc_grad_elbo(Q, model, data, advi.n_samples)
        δ = calc_step(i, η, α, g, s)
        ζ .+= δ .* g
        μ_new, σ_new = sample_transform(q)(ζ)

        # Check if we should stop
        if curr_elbo < best_elbo
            p += 1
        else
            p = 0
            q_best = M((μ=μ_new, σ=σ_new))
            best_elbo = curr_elbo
        end
        prev_elbo = curr_elbo

        # Update the parameters of the variational distribution
        Q = M((μ=μ_new, σ=σ_new))

        # Update the logger
        update_logger!(advi.logger, [μ_new..., σ_new...], g, curr_elbo)
    end

    if advi.verbose > 0
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
