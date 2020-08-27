
using StatsBase, DistributedArrays, Random
abstract type VariationalAlgorithm end

#===============================================================================
                    ADVI type definition and functions
===============================================================================#

mutable struct ADVI
    n_samples::Int
    max_iter::Int
    verbose::Int
    batch_size::Int
    ϵ::Float64
    logger::VarInfLogger
end

function optimize(advi::ADVI, elbo::ELBO, q::M, model::D, data::AbstractArray,
                  α::Float64=-1.0, η::Float64=-1.0) where {M<:MeanField, D<:PhyloModel}

    # Transform parameters to real space
    ζ = sample_invtransform(q)(Distributions.params(q))
    N = 2*length(q)

    # Setup some parameters
    prev_elbo = -Inf
    best_elbo = -Inf
    q_best = q
    Q = q
    s = zeros(N)
    counter = 0
    batch_counter = 0

    # Temp
    μ_new = 0
    σ_new = 0
    g = 0
    curr_elbo = 0

    # Show user the batching situation
    if advi.verbose > 1
        println("Using minibatches of size ", advi.batch_size, ".")
    end

    # Iterate until convergence or maximum of iterations is reached.
    for i in 1:advi.max_iter

        try
        # Sample minibatch and convert back to distributed object
        batch = StatsBase.sample(data, advi.batch_size, replace=false)
        batch = distribute(collect(batch))
        batch_factor = size(data)[1] / advi.batch_size

        # Calculate the gradient and update parameters
        g = calc_grad_elbo(Q, model, batch, advi.n_samples, batch_factor)
        δ = calc_step(i, η, α, g, s)

        # Ass gradient step updates to the parameters
        # ζ .+= δ .* g
        ζ .+= [δ[1:length(q)]..., δ[1:length(q)]...] .* g
        μ_new, σ_new = sample_transform(Q)(ζ)

        # Update the parameters of the variational distribution
        Q = M((μ=μ_new, σ=σ_new))

        # Calculate the ELBO
        # curr_elbo = elbo(Q, model, batch, batch_factor)
        curr_elbo = elbo(Q, model, data)
        if advi.verbose > 1; println("Iteration ", i, ": ", curr_elbo); end

        # Update the best Q
        if curr_elbo > best_elbo
            q_best = M((μ=μ_new, σ=σ_new))
            best_elbo = curr_elbo
        end
        counter = i

        # TODO: add logger functionality
        # Update the logger
        update_logger!(advi.logger, [μ_new..., σ_new...], g, curr_elbo)

        # Check if should exit
        if abs(curr_elbo - prev_elbo) < advi.ϵ
            break
        end
        prev_elbo = curr_elbo

        catch
            println("Caught error.")
            update_logger!(advi.logger, [μ_new..., σ_new...], g, curr_elbo)
            break
        end

    end

    # TODO: This finishing message isn't correct
    if advi.verbose > 0
        if counter == advi.max_iter
            println("Maximum of  ", counter, " iterations reached.")
        else
            println("Convergence of ADVI after ", counter, " iterations.")
        end
        println("Final ELBO: ", best_elbo)
    end
    return q_best
end

function calc_step(i::Int, η::Float64, α::Float64, g::AbstractArray{T}, s::AbstractArray{T}) where T<:Real
    ϵ = eps()
    s = (i == 1) ? g.^2 : α .* g.^2 .+ (1-α) .* s
    δ_new = η .* i^(-1/2 + ϵ) ./ (1.0 .+ s.^(1/2))
    return δ_new
end

function rolling_average!(l, val, n=20)
    if length(l) >= n; popfirst!(l) end# Remove first
    push!(l, val) # Add onto end
    return mean(l)
end
