using DataFrames

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

function display(logger::VarInfLogger)
    # TODO: Show logger info in text in nice overview: probably a dataframe
end

function plot(logger::VarInfLogger)
    #TODO: Plot the parameters of the logger
end


#===============================================================================
                                ADVI experimental
===============================================================================#

mutable struct ADVI
    n_samples::Int          # Amount of Monte Carlo samples in gradient computation
    max_iter::Int           # Maximum amount of iterations in optimization
    verbose::Int            # Degree of output text during optimization
    patience::Int           # Tolerance for non-improving iterations
    batchsize::Int          # Size of minibatch for gradient calculation
    logger::VarInfLogger    # Logger for optimization process
end

function grad_elbo(q::M, model::D, data::AbstractArray{T}, factor=1.0) where {M<:MeanField, D<:Distribution, T<:Real}

    # Get variational parameters (real space)
    p_real = sample_invtransform(q)(Distributions.params(q))

    # Create standard normal variational distribution
    N = length(q.dists)
    n = M(N)
    p = sample_transform(q)(p_real)     # Variational params in parameter space
    Q = M(p)
    η = rand(n)                         # Sample standard normal
    ζ = Vector{Float64}(inv_elliptical(Q, η))            # Transform to real space
    transform = model_transform(model)
    θ = [transform(ζ)...]               # Model parameters

    function grad_logprior(θ::Vector{Float64})
        f(x) = begin
            m = ConstantRatesModel(model.prior, model.tree, x[1], x[2], x[3])
            return logprior(m)
        end
        return ForwardDiff.gradient(f, θ)
    end

    function grad_loglikelihood(θ::Vector{Float64})
        m = DuplicationLoss(model.tree, [θ[1]], [θ[2]], θ[3], maximum(data))
        return Beluga.gradient(m, data)
    end

    function grad_logdetjac(ζ::Vector{Float64})
        f(x) = begin
            return TransformVariables.transform_and_logjac(transform, x)[2]
        end
        return ForwardDiff.gradient(f, ζ)
    end

    function grad_invtransform(ζ::Vector{Float64})
        return [exp(ζ[1]), exp(ζ[2]), exp(-ζ[3])]
    end

    function grad_entropy()
        return Vector{Float64}([zeros(N)..., ones(N)...])
    end

    x::Vector{Float64} = ((grad_logprior(θ) .+ grad_loglikelihood(θ) .* factor) .* grad_invtransform(ζ) .+ grad_logdetjac(ζ))
    μ::Vector{Float64} = x
    ω::Vector{Float64} = x .* (η .* p[2])
    return Vector{Float64}([μ..., ω...]) .+ grad_entropy()
end

function calc_grad_elbo(q::M, model::D, data::AbstractArray{T}, N::Int=10, factor=1.0) where {M<:MeanField, D<:Distribution, T<:Real}

    n = dimension(sample_transform(q))
    r = zeros(n)
    for i in 1:N
        r .+= grad_elbo(q, model, data, factor)
    end
    r ./= N
    return r
end

function (advi::ADVI)(elbo::ELBO, q::M, model::D, data::AbstractArray{T}, η::Float64=1.0, α::Float64=.1, ϵ=0.1) where {M<:MeanField, D<:Distribution, T<:Real}

    # Transform parameters to real space
    ζ = sample_invtransform(q)(Distributions.params(q))
    N = Int(dimension(sample_transform(q)))

    # Setup some parameters
    prev_elbo = -Inf
    best_elbo = -Inf
    curr_elbo = 0
    counter = 0
    p = 0
    q_best = q
    Q = q
    s = zeros(N)
    batch_count = round(size(data)[1]/advi.batchsize)
    μ_new, σ_new = nothing, nothing

    # Iterate until objective function converges
    for i in 1:advi.max_iter

        # Calculate parameter updates on minibatches
        for batchindex in 1:batch_count

            # Subsample a minibatch
            indices = rand(1:size(data)[1], advi.batchsize)
            batch = data[indices, :]

            # Calculate the gradient and update parameters
            g = calc_grad_elbo(Q, model, batch, advi.n_samples, Float64(batch_count))
            δ = calc_step(i, η, α, g, s)
            ζ .+= δ .* g
            μ_new, σ_new = sample_transform(q)(ζ)

            # Update the parameters of the variational distribution
            Q = M((μ=μ_new, σ=σ_new))

            # Update the logger each gradient step
            batch_elbo = elbo(Q, model, batch) * batch_count
            update_logger!(advi.logger, [μ_new..., σ_new...], g, batch_elbo)
        end

        # Calculate the ELBO on full data
        curr_elbo = elbo(Q, model, data)
        if advi.verbose > 1; println("Iteration ", i, ": ", curr_elbo); end

        if abs(curr_elbo - prev_elbo) < ϵ
            println("Convergence after ", i, " iterations.")
            break
        end

        # Check if we should stop
        if curr_elbo < best_elbo
            p += 1
        else
            p = 0
            q_best = M((μ=μ_new, σ=σ_new))
            best_elbo = curr_elbo
        end
        prev_elbo = curr_elbo

        # Check if should exit loop
        if p >= advi.patience
            break
        end
        counter += 1
    end

    # Print out the final output if required
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
