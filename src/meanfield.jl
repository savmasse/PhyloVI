using Distributions
using BenchmarkTools
using TransformVariables

#===============================================================================
            Mean field variational family struct and functions
===============================================================================#
abstract type AbstractVariationalFamily end
abstract type MeanField <: AbstractVariationalFamily end

function Base.length(q::T) where T<:MeanField
    return length(q.dists)
end

function Distributions.entropy(q::T) where T<:MeanField
    r = 0.0
    for i in 1:length(q.dists)
        r += entropy(q.dists[i])
    end
    return r
end

function Distributions.rand(q::T) where T<:MeanField
    s = Vector{Real}() # Must be real to not break ForwardDiff (??)
    for d in q.dists
        push!(s, rand(d))
    end
    return s
end
Distributions.rand(q::T, N::Int) where T<:MeanField = [rand(q) for _ in 1:N]


#===============================================================================
                Mean field Gaussian overloaded implementations
===============================================================================#

mutable struct MeanFieldGaussian <: MeanField
    dists::Vector{Normal}
end
function MeanFieldGaussian(x::AbstractVector{T}) where T<:Real
    N = Int(length(x)/2)
    dists = [Normal(x[i], x[N+i]) for i in 1:N]
    return MeanFieldGaussian(dists)
end
function MeanFieldGaussian(n::NamedTuple)
    μ = n[:μ]
    σ = n[:σ]
    N = length(μ)
    dists = [Normal(μ[i], σ[i]) for i in 1:N]
    return MeanFieldGaussian(dists)
end

function MeanFieldGaussian(N::Int) # Create standard normal
    dists::Vector{Normal} = [Normal() for i in 1:N]
    return MeanFieldGaussian(dists)
end

# TODO: Refactor this? sample_transform does not convey what this function does.
# It transforms the variational parameters, does not actually involve samples...
function sample_transform(q::MeanFieldGaussian)
    N = length(q.dists)
    t = as((μ = as(Array, asℝ, N), σ = as(Array, asℝ₊, N)))
    return t
end
sample_transform(q::MeanFieldGaussian, ζ::AbstractArray{T}) where T<:Real = sample_transform(q)(ζ)
sample_invtransform(q::MeanFieldGaussian) = inverse(sample_transform(q))
sample_invtransform(q::MeanFieldGaussian, n::NamedTuple) = sample_invtransform(q)(n)

function Distributions.entropy(q::MeanFieldGaussian)
    K = length(q.dists)
    r = K/2 * (1 + log(2*pi))
    d::Vector{Float64} = [i.σ for i in q.dists]
    r += sum(log.(d))
    return r
end

function grad_entropy(q::MeanFieldGaussian)
    # Calculate the gradient of the entropy for the current values of q
    N = length(q)
    d::Vector{Float64} = [i.σ for i in q.dists]
    δσ::Vector{Float64} = 1 ./ d
    r = zeros(N)
    append!(r, δσ)
    return r
end

function Distributions.params(q::MeanFieldGaussian)
    n = length(q.dists)
    μ = Vector{Float64}(undef, n)
    σ = Vector{Float64}(undef, n)
    for i in 1:n
        m, s = Distributions.params(q.dists[i])
        μ[i] = m
        σ[i] = s
    end
    return (μ = μ, σ = σ)
end

function elliptical_standardization(q::MeanFieldGaussian, ζ::AbstractVector{T}) where T<:Real
    # Get parameters (with log-transformed σ)
    μ, σ = Distributions.params(q)
    # Return standardized parameters
    η = (1.0 ./ σ) .* (ζ .- μ)
    return η
end

function inv_elliptical(q::MeanFieldGaussian, η::AbstractVector{T}) where T<:Real
    μ, σ = Distributions.params(q)
    ζ::Vector{T} = η .* σ .+ μ
    return ζ
end

function chain_factor(q::MeanFieldGaussian)
    μ, σ = Distributions.params(q)
    return LinearAlgebra.Diagonal(σ)
end


### Below this is experimental stuff intended to replace some of the above   ###

function PhyloVI.asvector(q::MeanFieldGaussian)
    # Return vector of variational parameters
    # Note the order of parameters:  [μ1...μN, σ1...σN] !!!
    μ = Vector{Float64}()
    σ = Vector{Float64}()
    for d in q.dists
        push!(μ, d.μ)
        push!(σ, d.σ)
    end
    return [μ..., σ...]
end

function (q::MeanFieldGaussian)(x::Vector{T}) where T<:AbstractFloat
    return MeanFieldGaussian(x)
end

# TODO: this is deprecated now ???
function init_q(model, λ::V, μ::V, d::V, η::V, σ::V) where V<:AbstractFloat

    # Get model dimensions
    L = length(model)
    N = getwgdcount(model)
    T = 2L + N + 1

    # Create vectors
    _λ = repeat([λ], L)
    _μ = repeat([μ], L)
    _d = repeat([d], N)
    _η = η
    _σ = repeat([σ], T)

    return MeanFieldGaussian([_λ..., _μ..., _d..., _η, _σ...])
end
