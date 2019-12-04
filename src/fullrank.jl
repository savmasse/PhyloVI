
using Distributions
using LinearAlgebra
using Beluga

#===============================================================================
                Full rank variational family implementations
===============================================================================#

mutable struct FullRankGaussian <: MeanField
    dists::MvNormal
end

function FullRankGaussian(N::Int)
    dists = MvNormal(zeros(N), ones(N))
    return FullRankGaussian(dists)
end

function FullRankGaussian(n::NamedTuple)
    μ = n.μ
    L = n.L
    Σ = L * transpose(L)
    return FullRankGaussian(MvNormal(μ, Σ))
end

function Distributions.rand(q::FullRankGaussian)
    return rand(q.dists)
end

function Distributions.params(q::FullRankGaussian)
    μ = q.dists.μ
    Σ = cov(q.dists)
    return (μ=μ, Σ=Σ)
end

function Distributions.entropy(q::FullRankGaussian)
    return entropy(q.dists)
end

function elliptical_standardization(q::FullRankGaussian, ζ::AbstractVector{T}) where T<:Real
    μ = q.dists.μ
    M = cov(q.dists)
    L = LinearAlgebra.cholesky(M).U
    η = L^-1 * (ζ - μ)
    return η
end

function inv_elliptical(q::FullRankGaussian, η::AbstractVector{T}) where T<:Real
    μ = q.dists.μ
    M = cov(q.dists)
    L = LinearAlgebra.cholesky(M).U
    ζ = L * η + μ
    return ζ
end

function sample_transform(q::FullRankGaussian)
    N = length(q)
    M = N*(N-1)/2
    t = as((μ = as(Array, asℝ, N), L = CorrCholeskyFactor(M)))
    return t
end

function sample_invtransform(q::FullRankGaussian)
    return inverse(sample_transform(q))
end

function getL(q::FullRankGaussian)
    N = length(q)
    M = cov(q.dists)
    L = LinearAlgebra.cholesky(M).U
    return L
end


#===============================================================================
                    ELBO type definition and functions
===============================================================================#

function (elbo::ELBO)(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}
    r = 0.0
    N = elbo.n_samples
    transform = model_transform(model)

    for i in 1:N
        ζ = rand(q)
        θ = transform(ζ)
        m = ConstantRatesModel(model.prior, model.tree, θ...)
        r += logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2]
    end

    return r/N - entropy(q)
end


function grad_elbo(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}

    # Get variational parameters (real space)
    N = length(q.dists)
    p = Distributions.params(q)
    L = getL(q)
    p = (μ = p.μ, L = L)
    p_real = sample_invtransform(q)(p)

    # Create standard normal variational distribution
    p = sample_transform(q)(p_real)     # Variational params in parameter space
    Q = M(p)
    η = rand(N)                         # Sample standard normal
    ζ = inv_elliptical(Q, η)            # Transform to real space
    transform = model_transform(model)
    θ = transform(ζ)

    # Create vector of theta
    θ = [i for i in Iterators.flatten(θ)]

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
        return [zeros(N)..., transpose(L^-1)...]
    end

    x = ((grad_logprior(θ) .+ grad_loglikelihood(θ)) .* grad_invtransform(ζ) .+ grad_logdetjac(ζ))
    ∇μ = x
    ∇L = (x * transpose(η))
    #println(x, ", ", η, ", ",  ∇L)
    return (Float64[∇μ..., ∇L...] .+ grad_entropy())
end

function calc_grad_elbo(q::FullRankGaussian, model::D, data::AbstractArray{T}, N::Int=10) where {D<:Distribution, T<:Real}

    n = length(q)
    r = zeros(Int(n + n*n))

    for i in 1:N
        r .+= grad_elbo(q, model, data)
    end
    r ./= N

    return r
end

function (advi::ADVI)(elbo::ELBO, q::FullRankGaussian, model::D, data::AbstractArray{T}, η::Float64=1.0, α::Float64=.1) where {D<:Distribution, T<:Real}

    r_indices = [1, 4, 6]
    # Transform parameters to real space
    p = Distributions.params(q)
    L = getL(q)
    p = (μ = p.μ, L = L)
    ζ = sample_invtransform(q)(p)
    N = Int(length(q))

    # Setup some parameters
    prev_elbo = -Inf
    best_elbo = -Inf
    counter = 0
    p = 0
    q_best = q
    Q = q
    #s = zeros(Int(N*(N+1)/2))
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
        indices = [1, 5, 9]
        g = calc_grad_elbo(Q, model, data, advi.n_samples)[indices]
        δ = calc_step(i, η, α, g, s)
        ζ[r_indices] .+= δ .* g
        μ_new, σ_new = sample_transform(q)(ζ)

        # Check if we should stop
        if curr_elbo < best_elbo
            p += 1
        else
            p = 0
            q_best = FullRankGaussian((μ=μ_new, L=σ_new))
            best_elbo = curr_elbo
        end
        prev_elbo = curr_elbo

        # Update the parameters of the variational distribution
        Q = FullRankGaussian((μ=μ_new, L=σ_new))

        # Update the logger
        update_logger!(advi.logger, [μ_new..., σ_new...], g, curr_elbo)
    end

    if advi.verbose > 0
        println("Finished ADVI after ", counter, " iterations.")
        println("Final ELBO: ", best_elbo)
    end
    return q_best
end

#===============================================================================
                        Main program for testing
===============================================================================#

# Get some example data
tree, _ = Beluga.example_data1()
Beluga.set_constantrates!(tree)
df = DataFrame(:A=>[4,3,1,4,3],:B=>[2,1,3,2,2],:C=>[4,4,1,1,3],:D=>[3,3,1,5,2])
data = profile(tree, df)

true_params = [0.1, 0.2, 0.3, 0.8]
λ_init, μ_init, q_init, η_init = 0.5, 0.5, 0.0, 0.5

prior = ConstantRatesPrior(Exponential(1.), Exponential(1.), Beta(1., 1.), Beta(6., 2.))
model = ConstantRatesModel{Float64}(prior, tree, λ_init, μ_init, η_init)
mv = MvNormal(repeat([0.], 3), repeat([1.], 3))
q = FullRankGaussian(mv)
Q = MeanFieldGaussian([Normal(0.1, 0.1) for i in 1:3])
L = length(q)
elbo = ELBO(1)
elbo(q, model, data)

m = cov(mv)
C = LinearAlgebra.cholesky(m)
grad_elbo(q, model, data)
calc_grad_elbo(q, model, data)

advi = ADVI(1, 20, 2, 100, VarInfLogger(Vector(), Vector(), Vector(), DataFrame(repeat([Float64[]], 5L+1))))
res = advi(elbo, q, model, data, 1e-12, 0.1)
res = Distributions.params(res)[:μ]
res = model_transform(model)(res)

p = Distributions.params(q)
p = (μ = p.μ, L = getL(q))
p_real = sample_invtransform(q)(p)
println(res)

# Make some plots of parameters
df = advi.logger.df
p = []
for i in 1:3
    push!(p, Plots.plot(df[i], legend=:none))
end
push!(p, Plots.plot(df[end], legend=:none))
Plots.plot(p...)
