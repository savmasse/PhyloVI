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
    s = Vector{Real}()
    for d in q.dists
        push!(s, rand(d))
    end
    return s
end


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
function sample_transform(q::MeanFieldGaussian)
    N = length(q.dists)
    t = as((μ = as(Array, asℝ, N), σ = as(Array, asℝ₊, N)))
    return t
end
sample_invtransform(q::MeanFieldGaussian) = inverse(sample_transform(q))

function Distributions.entropy(q::MeanFieldGaussian)
    K = length(q.dists)
    r = K/2 * (1 + log(2*π))
    d::Vector{Real} = [i.σ for i in q.dists]
    r += sum(log.(d))
    return r
end

function grad_entropy(q::MeanFieldGaussian)
    # Calculate the gradient of the entropy for the current values of q
    N = Int(length(q.dists)/2)
    d::Vector{Float64} = [i.σ for i in q.dists]
    δσ::Vector{Float64} = 1 ./ d
    r = zeros(N)
    append!(r, δσ)
    return r
end

function Distributions.params(q::MeanFieldGaussian)
    n = length(q.dists)
    μ = Vector{Real}(undef, n)
    σ = Vector{Real}(undef, n)
    for i in 1:n
        m, s = Distributions.params(q.dists[i])
        μ[i] = m
        σ[i] = s
    end
    return (μ = μ, σ = σ)
end

function elliptical_standardization(q::MeanFieldGaussian, ζ::AbstractVector{T}) where T<:Real
    # Get parameters (with log-transformed σ)
    μ, σ = params(q)
    # Return standardized parameters
    η = (1.0 ./ σ) .* (ζ .- μ)
    return η
end

function inv_elliptical(q::MeanFieldGaussian, η::AbstractVector{T}) where T<:Real
    μ, σ = Distributions.params(q)
    ζ::Vector{T} = η .* σ .+ μ
    return ζ
end


#===============================================================================
                Full rank variational family implementations
===============================================================================#

mutable struct FullRankGaussian <: MeanField
    dists::MvNormal
end
