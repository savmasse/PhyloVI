
using Distributions

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
        θ = transform(ζ)                    # Parameter

        # Normalize parameters
        ϕ = θ[:ϕ]
        N = 100
        for i in 1:N
            ϕ[i,:] ./= sum(ϕ[i,:])
        end
        ϕ .= max.(ϕ, eps())
        ϕ .= min.(ϕ, 1 - eps())
        # #println("Iteration ", i, ": ")
        # #println(θ[:ϕ])
        # try
        #     ζ = model_invtransform(model)(θ)
        # catch e
        #     println(θ)
        # end
        # ζ[findall(iszero, ζ)] .= eps()

        m = GaussianMixtureModel{T}(θ)
        r += logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2]
    end

    return r/N + entropy(n)
end

function grad_elbo(q::M, model::GaussianMixtureModel{T}, data::AbstractVector{T}) where {M<:MeanField, T<:Real}

    # Get variational parameters (real space)
    p = [i for i in Iterators.flatten(collect(params(q)))]
    p_real = sample_invtransform(q)(params(q))

    # Create standard normal variational distribution
    n = M(length(q.dists))

    function f(x)
        # Sample the variational distribution
        p = sample_transform(q)(x)          # Variational params in parameter space
        Q = M(p)
        η = rand(n)                         # Sample standard normal
        ζ = inv_elliptical(Q, η)            # Transform to real space
        ζ[findall(iszero, ζ)] .= eps()
        transform = model_transform(model)
        θ = transform(ζ)                    # Transform to parameter space

        # Normalize parameters
        ϕ = θ[:ϕ]
        N = 100
        for i in 1:N
            ϕ[i,:] ./= sum(ϕ[i,:])
        end
        # ϕ .= max.(ϕ, eps())
        # ϕ .= min.(ϕ, 1 - eps())
        # ζ = model_invtransform(model)(θ)
        # ζ[findall(iszero, ζ)] .= eps()

        m = GaussianMixtureModel{T}(θ)
        return logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2] + entropy(n)
    end
    g(x) = ForwardDiff.gradient(f, x)

    return g(p_real)
end


function (advi::ADVI)(elbo::ELBO, q::MeanFieldGaussian, model::D, data::AbstractVector{T}, η::Float64=1.0, α::Float64=.1) where {D<:Distribution, T<:Real}

    # Transform parameters to real space
    ζ = sample_invtransform(q)(params(q))
    N = Int(dimension(sample_transform(q)))

    # Setup some parameters
    prev = 0
    counter = 0
    patience = 0
    q_best = q
    Q = q
    s = zeros(N)
    PATIENCE = 5        # TODO: this should not be hardcoded!
    VERBOSE = 1         # TODO: this should not be hardcoded!

    # Enter convergence loop
    for i in 1:advi.max_iter

        # Check if should exit loop
        if patience >= PATIENCE
            break
        end
        counter += 1

        # Calculate the ELBO
        curr = elbo(Q, model, data)
        if VERBOSE == 1; println("Iteration ", i, ": ", curr); end

        # Calculate the gradient and update parameters
        g = calc_grad_elbo(Q, model, data, advi.n_samples)
        δ = calc_step(i, η, α, g, s)
        ζ .+= δ .* g
        ζ[findall(iszero, ζ)] .= eps()
        μ_new, σ_new = sample_transform(q)(ζ)

        # Normalize the categorical variables
        L = 100
        for i in 5:5+L-1
            n = (μ_new[i] + μ_new[i+L])
            μ_new[i] /= n
            μ_new[i+L] /= n
        end
        μ_new .= max.(μ_new, eps())
        μ_new .= min.(μ_new, 1 - eps())
        σ_new .= max.(σ_new, eps())

        # Check if we should stop
        if curr < prev
            patience += 1
        else
            patience = 0
            q_best = MeanFieldGaussian((μ=μ_new, σ=σ_new))
        end
        prev = curr

        # Update variational parameters
        Q = MeanFieldGaussian((μ=μ_new, σ=σ_new))
    end

    println("Finished ADVI after ", counter, " iterations.")
    return q_best
end


#===============================================================================
                        Main program for testing GMM
===============================================================================#

# Set true parameters
K = 2
μ = [-5., 5.]               # True mixture means
σ = [2., 2.]                # True standard deviations
n = 50
N = K*n

# Generate data and set labels
ϕ_true = zeros(N, K)             # True class labels
data = Vector{Float64}()
for i in 1:K
    ϕ_true[(i-1)*n+1:i*n, i] = ones(n)
    d = rand(Normal(μ[i], σ[i]), n)
    append!(data, d)
end

# Create a GMM and variational distribution
model = GaussianMixtureModel(μ, σ, ϕ_true)
ϕ_init = transpose(rand(Dirichlet(rand(2)), N))
q = MeanFieldGaussian([Normal(2., 1.) for i in 1:(2*N+2*K)])
p = params(q)
p.μ[2*K+1:end] = ϕ_init
init = [p.μ[1:(2*K)]..., reshape(ϕ_init, 200)..., 2 .*ones(2K + 2N)...]

# Create ELBO and ADVI objects
elbo = ELBO(50)
advi = ADVI(10, 100)

res = advi(elbo, q, model, data, .1, 0.9)
p = params(res)[:μ]

θ = model_transform(model)(p)
println(θ[:ϕ])
ϕ = θ[:ϕ]
for i in 1:N
    ϕ[i,:] ./= sum(ϕ[i,:])
end
display(ϕ)
