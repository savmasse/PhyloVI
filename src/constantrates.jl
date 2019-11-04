
using Distributions
using Beluga
using DataFrames
using Parameters
using TransformVariables
using ForwardDiff

#===============================================================================
                        Constant rates model definition
===============================================================================#

mutable struct ConstantRatesModel{T} <: ContinuousMultivariateDistribution where T<:Real
    prior::ConstantRatesPrior
    tree::SpeciesTree
    λ::T
    μ::T
    q::T
    η::T
end
function ConstantRatesModel(prior::ConstantRatesPrior, tree::SpeciesTree, λ::T, μ::T, q::T, η::T) where T<:Real
    return ConstantRatesModel{T}(prior, tree, λ, μ, q, η)
end

function logprior(d::ConstantRatesModel)
    λ=d.λ
    μ=d.μ
    q=d.q
    η=d.η
    @unpack dλ, dμ, dq, dη = d.prior
    lp  = logpdf(dλ, λ) + logpdf(dμ, μ) + logpdf(dη, η)
    lp += sum(logpdf.(dq, q))
    return lp
end

function Distributions.loglikelihood(d::ConstantRatesModel, data::AbstractArray{T, 2}) where T<:Real
    tree = d.tree
    λ = repeat([d.λ], length(tree))
    μ = repeat([d.μ], length(tree))
    d_model = DuplicationLoss(tree, λ, μ, d.η, maximum(data))
    println(d_model)
    return Beluga.logpdf(d_model, data)
end

function model_transform(d::ConstantRatesModel)
    t = as((λ = as_unit_interval, μ = as_unit_interval, q = as_unit_interval, η = as_unit_interval))
    return t
end

function model_invtransform(d::ConstantRatesModel)
    t = inverse(model_transform(d))
    return t
end

function update_params!(d::ConstantRatesModel, θ::NamedTuple) where T<:Real
    d.prior = ConstantRatesPrior(θ...)
end


#===============================================================================
                        Overloaded ELBO functions
===============================================================================#

function (elbo::ELBO)(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}
    r = 0.0
    N = elbo.n_samples
    n = M(length(q.dists))

    for i in 1:N
        η = rand(n)                         # Standardized parameters
        ζ = inv_elliptical(q, η)            # Real space parameters
        transform = model_transform(model)
        θ = transform(ζ)                    # Parameter space
        m = ConstantRatesModel(model.prior, model.tree, θ...)
        r += logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2]
    end

    return r/N - entropy(q)
end

function grad_elbo(q::M, model::D, data::AbstractArray{T}) where {M<:MeanField, D<:Distribution, T<:Real}

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
        transform = model_transform(model)
        θ = transform(ζ)                    # Transform to parameter space
        #println(model)
        m = ConstantRatesModel(model.prior, model.tree, θ...)
        return logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(transform, ζ)[2] - entropy(Q)
    end
    g(x) = ForwardDiff.gradient(f, x)

    return g(p_real)
end

function calc_grad_elbo(q::M, model::D, data::AbstractArray{T}, N::Int=10) where {M<:MeanField, D<:Distribution, T<:Real}

    n = dimension(sample_transform(q))
    r = zeros(T, n)
    for i in 1:N
        r .+= grad_elbo(q, model, data)
        println("Gradient iteration ", i , " finished.")
    end
    r ./= N
    return r
end


function (advi::ADVI)(elbo::ELBO, q::M, model::D, data::AbstractArray{T}, η::Float64=1.0, α::Float64=.1,
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


#===============================================================================
                            Main program  for testing
===============================================================================#

# Get some example data
tree, _ = Beluga.example_data1()
Beluga.set_constantrates!(tree)
df = DataFrame(:A=>[4,3,1,4,3],:B=>[2,1,3,2,2],:C=>[4,4,1,1,3],:D=>[3,3,1,5,2])
data = profile(tree, df)

true_params = [0.1, 0.2, 0.3, 0.8]
λ_init = 0.5
μ_init = 0.5
q_init = 0.5
η_init = 0.5

prior = ConstantRatesPrior(Exponential(1.), Exponential(1.), Beta(1., 1.), Beta(6., 2.))
model = ConstantRatesModel{Float64}(prior, tree, λ_init, μ_init, q_init, η_init)
Q = MeanFieldGaussian([Normal(0.1, 0.1) for i in 1:4]) # Initial mean-field distribution

elbo = ELBO(10)
elbo(Q, model, data)

advi = ADVI(10, 10)
grad_elbo(Q, model, data)
calc_grad_elbo(Q, model, data)

function foo(x)
    m = ConstantRatesModel{Real}(model.prior, model.tree, x...)
    return loglikelihood(m, data)
end
g(x) = ForwardDiff.gradient(foo, x)

g(rand(4))
