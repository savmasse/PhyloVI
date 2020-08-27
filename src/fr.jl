
using Distributions
using LinearAlgebra
using Beluga
using Plots
using CSV
using Measures
using PhyloVI

include("../src/vimodel.jl")
include("../src/phylomodel.jl")
include("../src/wgdbelugamodel.jl")
include("../src/meanfield.jl")
include("../src/elbo.jl")
include("../src/logger.jl")
include("../src/advi.jl")

# Set seed for reproducibility
Random.seed!(1)

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
    return FullRankGaussian(MvNormal(μ, L*transpose(L)))
end

function Distributions.rand(q::FullRankGaussian)
    return rand(q.dists)
end

function Distributions.params(q::FullRankGaussian)
    μ = q.dists.μ
    Σ = cov(q.dists)
    L = LinearAlgebra.cholesky(Σ).L
    return (μ=μ, L=L)
end

function Distributions.entropy(q::FullRankGaussian)
    return entropy(q.dists)
end

function elliptical_standardization(q::FullRankGaussian, ζ::AbstractVector{T}) where T<:Real
    _, L = sample_transform(q, )
    η = L^-1 * (ζ - μ)
    return η
end

function inv_elliptical(q::FullRankGaussian, η::AbstractVector{T}) where T<:Real
    μ, L = Distributions.params(q)
    ζ = L * η + μ
    return ζ
end

function sample_transform(q::FullRankGaussian)
    N = length(q)
    t = as((μ = as(Array, asℝ, N), L = CorrCholeskyFactor(N+1)))
    return t
end

function sample_transform(q::FullRankGaussian, ζ::AbstractVector{T}) where T<:Real
    # μ, L = sample_transform(q)(ζ)
    # L = collect(transpose(L)) # Need to return lower diagonal matrix
    # return ((μ=μ, L=L))
    d = length(q)
    μ = ζ[1:d]
    z = copy(ζ[d+1:end])
    L = zeros(d, d)
    for i in 1:d
        for j in i:d
            L[j, i] = popfirst!(z)
        end
    end
    return ((μ=μ, L=L))
end

function sample_invtransform(q::FullRankGaussian)
    return inverse(sample_transform(q))
end
function sample_invtransform(q::FullRankGaussian, n::NamedTuple)
    # Transform expects an UpperTriangular matrix
    # μ, L = n
    # L = UpperTriangular(collect(transpose(L)))
    # nt = ((μ=μ, L=L))
    # return sample_invtransform(q)(nt)
    μ, L = n
    ζ = [L[j, i] for i in 1:d for j in i:d]
    return [μ..., ζ...]::Vector{Float64}
end

function getL(q::FullRankGaussian)
    N = length(q)
    M = cov(q.dists)
    L = LinearAlgebra.cholesky(M).L
    return L
end

function chain_factor(q::FullRankGaussian)
    N = length(q)
    return LinearAlgebra.I(N)
end

#===============================================================================
                    ELBO type definition and functions
===============================================================================#

function (elbo::ELBO)(q::Q, model::M, data::PhyloData, batch_factor::Float64=1.0) where {Q<:MeanField, M<:PhyloModel}

    # Take N real-space samples from the variational distribution and transform
    # to parameter space.
    N = elbo.n_samples
    ζ = rand(q, N)
    T = model_transform(model)
    θ = [T(ζ[i]) for i in 1:N]

    # Take ELBO N times and take mean
    r = 0.0
    for i in 1:N
        m = model(θ[i])
        r += logprior(m) + loglikelihood(m, data) * batch_factor
    end
    return r/N + entropy(q)
end

function calc_grad_elbo2(q::FullRankGaussian, model::WGDBelugaModel, data, N::Int=10, batch_factor=1.0) where T<:Real
    n = length(q)
    r = [zeros(n), zeros(n, n)]
    for i in 1:N
        r .+= grad_elbo(q, model, data, batch_factor)
    end
    r ./= N
    return r
end

function grad_elbo(q::FullRankGaussian, model::M, data::AbstractArray, batch_factor::Float64=1.0) where M<:PhyloModel

    # Check inputs
    if batch_factor < 1.0
        throw("Batch factor should be in domain [1.0; ∞[")
    end

    # Sample unit variational function and transform
    u, L = Distributions.params(q)
    n = FullRankGaussian(length(q))
    η = rand(n)
    ζ = inv_elliptical(q, η)
    t = model_transform(model)
    θ = t(ζ)
    m = model(θ)

    # Put the components together
    γ = (grad_logprior(m) .+ grad_loglikelihood(m, data) .* batch_factor) .+ grad_logdetjac(m, ζ)
    μ = γ
    ω = tril(γ * transpose(η)) .+ inv(diagm(diag((L))))

    return sample_invtransform(q, (μ=μ, L=ω))
end

function optimize(advi::ADVI, elbo::ELBO, q::FullRankGaussian, model::D, data::AbstractArray,
                  α::Float64=-1.0, η::Float64=-1.0) where D<:PhyloModel

    # Transform parameters to real space
    ζ = [q.dists.μ..., vec(getL(q).data)...]
    T = length(q)
    N = T^2 + T

    # Setup some parameters
    best_elbo = -Inf
    q_best = q
    Q = q
    s = zeros(T)
    sμ = zeros(T)
    sL = zeros(T, T)
    μ = q.dists.μ
    L = getL(q).data
    avg = Float64[]

    # Show user the batching situation
    if advi.verbose > 1
        println("Using minibatches of size ", advi.batch_size, ".")
    end

    for i in 1:advi.max_iter

        # Sample minibatch and convert back to distributed object
        batch = StatsBase.sample(data, advi.batch_size, replace=false)
        batch = distribute(collect(batch))
        batch_factor = size(data)[1] / advi.batch_size

        # Calculate the gradient and update parameters
        g = calc_grad_elbo2(Q, model, batch, advi.n_samples, batch_factor)
        ρμ = calc_step(i, η, α, g[1], sμ)
        ρL = calc_step(i, η, α, g[2], sL)

        # Update the real-space parameters
        μ .+= ρμ .* g[1]
        # L .+= diagm(diag(ρL)) .* g[2]
        L .+= diagm(ρμ) * g[2]
        μ_new, Σ_new = transform_fullrank(μ, L)

        # Update the parameters of the variational distribution
        Q = FullRankGaussian((μ=μ_new, Σ=Σ_new))

        # Calculate the ELBO
        curr_elbo = elbo(Q, model, data)
        # curr_elbo = elbo(Q, model, batch)
        rolling_elbo = rolling_average!(avg, curr_elbo, 20)
        if advi.verbose > 1; println("Iteration ", i, ": ", curr_elbo, " (avg: ", rolling_elbo, ")"); end

        # Update result if better than previous best
        if curr_elbo > best_elbo
            best_elbo = curr_elbo
            q_best = Q
            println("Found new best: ", curr_elbo)
        end

        # Log the new values
        push!(advi.logger.df, [μ..., vec(L)...])
    end

    return q_best
end

function transform_fullrank(μ, L)
    Σ = L * transpose(L)
    return (μ=μ, Σ=Σ)
end

function inv_transform_fullrank(μ, Σ)
    L = LinearAlgebra.cholesky(Σ).L
    return (μ=μ, L=L)
end

function PhyloVI.asvector(L::LowerTriangular)
    v = Float64[]
    for i in 1:size(L)[1]
        for j in 1:i
            push!(v, L[i, j])
        end
    end
    return v
end

function rolling_average!(l, val, n=20)
    if length(l) >= n; popfirst!(l) end# Remove first
    push!(l, val) # Add onto end
    return mean(l)
end

#===============================================================================
                        Quick test of functionality
===============================================================================#
#
# datadir = "./PhyloVI/data/"
# tree = open(joinpath(datadir, "species_trees/plants2.nw"), "r") do f ; readline(f); end
# df = CSV.read(joinpath(datadir, "branch_wise/1.counts.csv"), delim=",")
#
# # Init model
# λ, μ, η = 1., 1., 0.5
# model, profile = DLWGD(tree, df, λ, μ, η)
#
# # Simulate some more data
# rr = CSV.read(".\\PhyloVI\\data\\branch_wise\\1.rates.csv", delim=",")
# η = 0.85
# x = [rr[:λ]..., rr[:μ]..., η]
# model = model(x)
# df = rand(model, 1000)
# model, profile = DLWGD(tree, df, λ, μ, η)
#
# data = profile
# prior = IidRevJumpPrior(
#     Σ₀=[0.5 0.45 ; 0.45 0.5],
#     X₀=MvNormal(log.(ones(2)), [0.5 0.45 ; 0.45 0.5]),
#     πK=DiscreteUniform(0,20),
#     πq=Beta(1,1),
#     πη=Beta(3,1),
#     Tl=treelength(model))
#
# # Create a Beluga model with a WGD
# bm = WGDBelugaModel(prior, model)
# N = length(Beluga.getwgds(model))
# L = length(model) - 2N
# T = 2*L + N + 1
#
#
# ##### Full rank specific stuff below
#
# # Run ADVI algorithm
# μ = 0.0; σ = 0.1
# μ = repeat([μ], 34)
# σ = repeat([σ], 34)
# push!(μ, 1.)
# push!(σ, 0.05)
# f = FullRankGaussian(MvNormal(μ, σ))
#
# elbo = ELBO(5)
# advi = ADVI(1, 500, 2, 100, 1e-2, VarInfLogger(DataFrame(repeat([Float64[]],T^2 + T))))
# Q = optimize(advi, elbo, f, bm, data, 0.1, .04)
#
# # Process results
# r = Distributions.params(Q)[:μ]
# res = model_transform(bm)(r)
# rr = CSV.read(".\\PhyloVI\\data\\branch_wise\\1.rates.csv", delim=",")
# d = DataFrame((λ_truth=rr[:λ], λ=res[:λ], μ_truth=rr[:μ], μ=res[:μ]))
#
# # Quick MCMC run for comparison
# # ch = RevJumpChain(data=data, model=deepcopy(model), prior=prior)
# # init!(ch)
# # rjmcmc!(ch, 1000, show=10, trace=1)
#
# # Sample the variational distribution create histogram
# p = hcat(rand(Q, 10_000)...)
# plots = []
# for i in 1:34
#     # x = Plots.histogram(exp.(transpose(p)[:, i]),
#     x = StatsPlots.plot(LogNormal(Q.dists.μ[i], sqrt(Q.dists.Σ.mat[i, i])),
#                         alpha=1.,
#                         legend=:none,
#                         margin=0mm,
#                         tickfont=font(3),
#                         normalize=true,
#                         color=:lightblue,
#                         fill=true)
#
#     if i < L+1
#         Plots.vline!([d[:λ_truth][i]])
#     else
#         Plots.vline!([d[:μ_truth][i-L]])
#     end
#     push!(plots, x)
# end
# Plots.plot(plots...,fmt=:svg)
#
#
# pl = []
# for i in 1:17
#     t = Plots.plot(exp.(advi.logger.df[i]),
#                     legend=:none,
#                     tickfont=font(4))
#     push!(pl, t)
# end
# Plots.plot(pl...)
#
# # Compare the MCMC with the full rank gaussian
# n = [Symbol(string("λ", i)) for i in 1:L]
# append!(n, [Symbol(string("μ", i)) for i in 1:L])
# push!(n, Symbol("η"))
# temp = names(advi.logger.df)
# temp[1:T] = n
# names!(advi.logger.df, temp)
# all_rates = [Symbol(string("λ", i)) for i in 1:L]
# compare_methods(Q, bm, advi.logger, ch.trace, all_rates, x, 100)
#
#
# # Create some 2D plots to view correlation between μ and λ in the same branch
# plots = []
# for i in 1:17
#     a = ch.trace[Symbol(string("λ", i))]
#     b = ch.trace[Symbol(string("μ", i))]
#     c1 = [ (a[i], b[i]) for i in 1:length(a) ]
#     pt = Plots.scatter(c1, label="MCMC", margin=0mm, legend=:none, tickfont=font(4), alpha=0.8, markersize=2, color=:black)
#
#     p = hcat(rand(Q, length(a))...)
#     d = exp.(p[i, :])
#     e = exp.(p[L+i, :])
#     c2 = [ (d[i], e[i]) for i in 1:length(a) ]
#     Plots.scatter!(c2, label="ADVI", alpha=0.8, markersize=2, color=:yellow)
#
#     push!(plots, pt)
# end
# Plots.plot(plots...)
