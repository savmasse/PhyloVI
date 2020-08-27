using Plots
using StatsPlots
using Distributions
using ForwardDiff

#===============================================================================
                        Define Gaussian Mixture Model
===============================================================================#

mutable struct GaussianMixtureModel{T} <: ContinuousMultivariateDistribution where T<:Real
    μ::Vector{T}
    σ::Vector{T}
    ϕ::Array{T}
end
function GaussianMixtureModel{T}() where T<:Real
    μ = Vector{T}()
    σ = Vector{T}()
    ϕ = Array{T, 1}()
    return GaussianMixtureModel(μ, σ, ϕ)
end
function GaussianMixtureModel{T}(n::NamedTuple) where T<:Real
    μ = n[:μ]
    σ = n[:σ]
    ϕ = n[:ϕ]
    # ϕ .= max.(ϕ, eps())
    # ϕ .= min.(ϕ, 1.0-eps())
    # for i in 1:N
    #     ϕ[i,:] ./= sum(ϕ[i,:])
    # end
    return GaussianMixtureModel(μ, σ, ϕ)
end
function GaussianMixtureModel{T}(x::AbstractVector{V}) where {T<:Real, V<:Real}
    μ = x[1:2]
    σ = x[3:4]
    N = Int(length(x[5:end])/2)
    ϕ = reshape(x[5:end], N, 2)
    # ϕ .= max.(ϕ, eps())
    # ϕ .= min.(ϕ, 1.0-eps())
    # for i in 1:N
    #     ϕ[i,:] ./= sum(ϕ[i,:])
    # end
    return GaussianMixtureModel(μ, σ, ϕ)
end

function logprior(gmm::GaussianMixtureModel)
    return 0    # Hardcoded for now, so no prior...
end
function Distributions.pdf(gmm::GaussianMixtureModel, x::AbstractVector{T}) where T<:Real
    K = length(gmm.μ)
    n::Vector{Normal} = [Normal(gmm.μ[i], gmm.σ[i]) for i in 1:K]

    r = Vector()
    for i in eachindex(x)
        s = 0.0
        for k in 1:K
            s += gmm.ϕ[i, k] * pdf(n[k], x[i])
        end
        push!(r, s)
    end
    return r
end
Distributions.logpdf(gmm::GaussianMixtureModel, x::AbstractVector{T}) where T<:Real = return log.(pdf(gmm, x))
function Distributions.loglikelihood(gmm::GaussianMixtureModel, x::AbstractVector)
    r = sum(logpdf(gmm, x))
    return r
end
function Distributions.params(gmm::GaussianMixtureModel)
    n = (μ = gmm.μ, σ = gmm.σ, ϕ = gmm.ϕ)
    return n
end
function model_transform(gmm::GaussianMixtureModel)
    K = length(gmm.μ)
    N = size(gmm.ϕ)[1]
    t = as((μ = as(Array, asℝ, K), σ = as(Array, asℝ₊, K), ϕ = as(Array, as_unit_interval, N, K)))
    return t
end
function model_invtransform(gmm::GaussianMixtureModel)
    t = model_transform(gmm)
    return inverse(t)
end

function (elbo::ELBO)(q::M, model::GaussianMixtureModel{T}, data::AbstractVector{T}) where {M<:MeanField, T<:Real}
    r = 0.0
    N = elbo.n_samples
    n = M(length(q.dists))

    for i in 1:N
        η = rand(n)                         # Standardized parameters
        ζ = inv_elliptical(q, η)            # Real space parameters
        ζ[findall(iszero, ζ)] .= eps()
        transform = model_transform(model)
        θ = transform(ζ)                    # Parameter space

        # Normalize parameters
        ϕ = θ[:ϕ]
        ϕ .= max.(ϕ, eps())
        ϕ .= min.(ϕ, 1.0-eps())
        # θ[:σ] .= max.(θ[:σ], eps())
        m = Int((length(q) - 2*length(model.μ))/2)
        for i in 1:m
            ϕ[i,:] ./= sum(ϕ[i,:])
        end
        ζ = model_invtransform(model)(θ)

        m = GaussianMixtureModel{T}(θ)
        r += logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2]
    end

    return r/N + entropy(n)
end

function grad_elbo(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}

    # Get variational parameters (real space)
    p_real = sample_invtransform(q)(Distributions.params(q))

    # Create standard normal variational distribution
    K = length(model.μ)
    N = length(q.dists) - 2K
    n = M(N+2K)
    p = sample_transform(q)(p_real)     # Variational params in parameter space
    Q = M(p)
    η = rand(n)                         # Sample standard normal
    ζ = inv_elliptical(Q, η)            # Transform to real space
    ζ[findall(iszero, ζ)] .= eps()
    transform = model_transform(model)
    θ = transform(ζ)

    # Normalize parameters
    ϕ = θ[:ϕ]
    ϕ .= max.(ϕ, eps())
    ϕ .= min.(ϕ, 1.0 - eps())
    # θ[:σ] .= max.(θ[:σ], eps())
    for i in 1:Int(N/2)
        ϕ[i,:] ./= sum(ϕ[i,:])
    end
    # ζ = model_invtransform(model)(θ)

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
        μ = ones(K)
        σ = exp.(ζ[K+1:2K])
        ϕ = zeros(N)
        for i in 1:N
            if ζ[2K+i] > 0
                ϕ[i] = exp(-ζ[2K+i])
            elseif ζ[2K+i] < 0
                ϕ[i] = exp(ζ[2K+i])
            else
                ϕ[i] = 1.0
            end
        end
        return [μ..., σ..., ϕ...]
    end
    function grad_entropy()
        return [zeros(2K+N)..., ones(2K+N)...]
    end

    x = ((grad_logprior(θ) .+ grad_loglikelihood(θ)) .* grad_invtransform(ζ) .+ grad_logdetjac(ζ))
    μ = x
    ω = x .* (η .* p[2])
    return Float64[μ..., ω...] .+ grad_entropy()
end

function (advi::ADVI)(elbo::ELBO, q::M, model::D, data::AbstractArray{T}, η::Float64=1.0, α::Float64=.1) where {M<:MeanField, D<:Distribution, T<:Real}

    # Transform parameters to real space
    ζ = sample_invtransform(q)(Distributions.params(q))
    N = Int(dimension(sample_transform(q)))

    # Setup some parameters
    prev_elbo = -Inf
    best_elbo = -Inf
    counter = 0
    p = 0
    q_best = q
    best_index = 0
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
        if abs(curr_elbo - prev_elbo) < .05
            p += 1
        else
            p = 0
            q_best = M((μ=μ_new, σ=σ_new))
            best_elbo = curr_elbo
            best_index = i
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
        println("Final best index: ", best_index)
    end
    return (Q, best_index)
end


#===============================================================================
                        Main program for testing GMM
===============================================================================#
# Set true parameters
K = 2
μ = [-10., 10.]                 # True mixture means
σ = [1., 1.]                    # True standard deviations
n = 50
N = K*n
# Generate data and set labels
ϕ_true = zeros(N, K)            # True class labels
data = Vector{Float64}()
for i in 1:K
    ϕ_true[(i-1)*n+1:i*n, i] = ones(n)
    d = rand(Normal(μ[i], σ[i]), n)
    append!(data, d)
end

# Create a GMM and variational distribution
model = GaussianMixtureModel(μ, σ, ϕ_true)
q = MeanFieldGaussian(2K + 2N)
p = Distributions.params(q)
init = [p.μ[1:(2*K)]..., repeat([0.], 200)..., ones(2K + 2N)...]
q = MeanFieldGaussian(init)

# Create ELBO and ADVI objects
elbo = ELBO(10)
logger = VarInfLogger(Vector(), Vector(), Vector(), DataFrame(repeat([Float64[]], 8K + 8N + 1)))
advi = ADVI(1, 200, 2, 10, logger)
res, index = advi(elbo, q, model, data, .5, .3)
ζ = Distributions.params(res)[:μ]

# Convert to correct probabilities
θ = model_transform(model)(ζ)
ϕ = θ.ϕ
for i in 1:N
    ϕ[i,:] ./= sum(ϕ[i,:])
end
m = GaussianMixtureModel{Float64}(θ)

# Make a plot
df = advi.logger.df
p = []
push!(p, Plots.plot(df[1], legend=:none))
push!(p, Plots.plot(df[2], legend=:none))
push!(p, Plots.plot(df[end], legend=:none))
Plots.plot(p...)

Plots.plot(df[30])
Plots.plot(df[130])
