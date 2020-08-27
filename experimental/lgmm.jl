using Beluga
using PhyloVI
using DataFrames
using BenchmarkTools
using Random

include("../src/vimodel.jl")
include("../src/phylomodel.jl")
include("../src/meanfield.jl")

#===============================================================================
                Define Gaussian Mixture Model with labels
===============================================================================#

struct LGMM{T} <: PhyloModel where T<:Real
    K::Int
    μ::Vector{T}
    σ::Vector{T}
    c::Vector{Int}
end

function (model::LGMM)(θ::Vector{T}) where T<:Real
    K = model.K
    μ = θ[1:K]
    σ = θ[K+1:2K]
    c = θ[2K+1:end]
    return LGMM{Float64}(K, μ, σ, c)
end

function (model::LGMM)(nt::NamedTuple)
    μ, σ, c =  nt
    return LGMM{Float64}(model.K, μ, σ, c)
end

function PhyloVI.asvector(model::LGMM)
    return [model.μ..., model.σ..., model.c...]
end

function params(model::LGMM)
    θ = asvector(model)
    K = model.K
    return (μ = θ[1:K], σ=θ[K+1:2K], c=θ[2K+1:end])
end

function logprior(model::LGMM)
    lp = sum(logpdf.(Normal(0, 6), model.μ))
    lp += sum(logpdf.(LogNormal(-1, .5), model.σ))
    lp += length(model.c) * log(1/model.K)
    return lp
end

function loglikelihood(model::LGMM, data::AbstractArray{T}) where T<:Real
    res = zero(T)
    for i in 1:length(data)
        c = model.c[i]
        res += logpdf(Normal(model.μ[c], model.σ[c]), data[i])
    end
    return res
end

function model_transform(model::LGMM)
    t = as((μ = as(Array, as_real, model.K),
            σ = as(Array, as_positive_real, model.K),
            c = as(Array, as_real, length(model.c))))
    return t
end

function model_invtransform(model::LGMM)
    t = inverse(model_transform(model))
    return t
end


#===============================================================================
                Define MeanField distributions with Categoricals
===============================================================================#

struct MeanFieldGMM <: MeanField
    K # Have to know how many clusters
    dists
end

function MeanFieldGMM(nt::NamedTuple)
    μ, σ, ϕ = nt
    K = Int(length(μ)/2)
    dists = Any[]
    for i in 1:2K
        push!(dists, Normal(μ[i], σ[i]))
    end
    for i in 1:length(ϕ)
        push!(dists, Categorical(ϕ[i]))
    end
    return MeanFieldGMM(K, dists)
end

function (q::MeanFieldGMM)(nt::NamedTuple)
    return MeanFieldGMM(nt)
end

function PhyloVI.asvector(q::MeanFieldGMM)
    p = Distributions.params(q)
    return collect(Iterators.flatten(p))
end

function Distributions.params(q::MeanFieldGMM)
    K = q.K
    μ = Float64[]
    σ = Float64[]
    for i in 1:2K # Two latent variables per cluster, each 2 variational params
        push!(μ, q.dists[i].μ)
        push!(σ, q.dists[i].σ)
    end

    # The rest are local latent variables for each sample
    ϕ = [q.dists[i].p for i in 2K+1:length(dists)]

    return (μ=μ, σ=σ, ϕ=ϕ)
end

function sample_transform(q::MeanFieldGMM)
    K = q.K
    N = length(dists) - 2K
    t = as((μ = as(Array, asℝ, 2K),
            σ = as(Array, asℝ₊, 2K),
            ϕ = as(Array, UnitSimplex(K), N)))
    return t
end
function sample_invtransform(q::MeanFieldGMM)
    return inverse(sample_transform(q))
end

Random.seed!(1)

# Test some things
K = 2
N = 500
μ = [-5., 5.]
σ = ones(K)

# Create some data
N1 = 250
N2 = 250
c = Int[ones(N1)..., 2*ones(N2)...]
data = rand(Normal(μ[1], σ[1]), N1)
append!(data, rand(Normal(μ[2], σ[2]), N2))
model = LGMM(K, μ, σ, c)

dists = Any[Normal(-0.1, .01) for _ in 1:2K]
append!(dists, [Categorical(K) for _ in 1:N])
q = MeanFieldGMM(K, dists)

ζ = rand(q)
@time logp(model, ζ, data)
@time logq(q, ζ)
@time ∇logq(q, ζ)
@time bbvi_estimator_cv(10, q, model, data)

using Flux.Optimise
Q, hist = optimize(1000, 5, q, model, ADAM(0.05), data)

# Check the results...
p = Distributions.params(Q)
μ = p[:μ][1:K]
σ = p[:μ][K+1:2K]
ϕ = p[:ϕ]
c = [argmax(p[:ϕ][i]) for i in 1:N]

println("μ: ", μ)
println("σ: ", σ)
println("Counts: ", counts(c))
Plots.plot(hist)

d1 = data[c .== 1]
d2 = data[c .== 2]
Plots.histogram(d1, bins = 50)
Plots.histogram!(d2, bins = 50)
