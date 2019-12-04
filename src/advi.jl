using DataFrames
using Plots

abstract type VariationalAlgorithm end

#===============================================================================
                            ADVI optimization logger
===============================================================================#

mutable struct VarInfLogger
    θ::Vector           # Parameters in variational parameter space
    ∇::Vector           # Gradients in real space
    objective::Vector   # Variational objective (ELBO or other)
    df::DataFrame       # DataFrame containing the results for each output
end

function update_logger!(logger::VarInfLogger, θ, ∇, objective)
    push!(logger.df, [θ..., ∇..., objective])
end

function plot(logger::VarInfLogger, model, index::Int)
    p = get_plot(logger, model, index)
    Plots.plot(p)
end

function plot_all(logger::VarInfLogger, model)
    N = Int((size(logger.df, 2) - 1) / 4)
    p = [get_plot(logger, model, i) for i in 1:N+1]
    Plots.plot(p...)
end

function get_plot(logger::VarInfLogger, model, index::Int)
    # Plot logger graph for given index
    N = Int((size(logger.df, 2) - 1) / 4)
    if index > N+1
        throw("Index out of range for model parameters in dataframe.")
    end

    p = nothing
    if index != N+1
        data = convert(Matrix, logger.df[:, 1:N])
        data = [collect(model_transform(model)(d))[index] for d in eachrow(data)]
        p = Plots.plot(data, ribbon=logger.df[!, index + N], title=names(logger.df)[index],
                    xlabel="iterations")
    else
        data = logger.df[!, end]
        p = Plots.plot(data, title="ELBO")
    end
    return p
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

function (advi::ADVI)(elbo::ELBO, q::M, model::D, data::AbstractArray{T}, η::Float64=1.0, α::Float64=.1, batch_factor::Float64=1.) where {M<:MeanField, D<:Distribution, T<:Real}

    # Transform parameters to real space
    ζ = sample_invtransform(q)(Distributions.params(q))
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

        # Get a minibatch if required
        batch = data
        if batch_factor != 1.0
            batch_size = Int(batch_factor*size(data)[1])
            indices = rand(1:size(data)[1], batch_size)
            batch = data[indices, :]
            # Fix size if data is 1D
            if length(size(data)) == 1; batch = data[indices]; end
        end

        # Calculate the gradient and update parameters
        g = calc_grad_elbo(Q, model, batch, advi.n_samples, 1.0/batch_factor)
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
        if counter == advi.max_iter
            println("Forced stop of ADVI after ", counter, " iterations.")
        else
            println("Convergence of ADVI after ", counter, " iterations.")
        end
        println("Final ELBO: ", best_elbo)
    end
    return q_best, Q
end

function calc_step(i::Int, η::Float64, α::Float64, g::AbstractVector{T}, s::AbstractVector{T}) where T<:Real
    ϵ = eps()
    s = (i == 1) ? g.^2 : α .* g.^2 .+ (1-α) .* s
    δ_new = η .* i^(-1/2 + ϵ) ./ (1.0 .+ s.^(1/2))
    return δ_new
end
