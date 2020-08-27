
# Mixture of Beluga models

struct BelugaMixture{T} <: PhyloModel where T<:Real
    K::Int
    mixtures
    π::Vector{T}
    prior

    # # Inner constructor in order to check inputs
    # function BelugaMixture(K::Int, mixtures::Vector{T}, π::Vector{Float64}) where T<:DLWGD
    #     if sum(π) != 1.0
    #         throw("Mixture coefficients do not sum to 1.")
    #     end
    #     if !(length(π) == length(mixtures) == K)
    #         throw("Lengths of arguments do not correspond with given value for K.")
    #     end
    #     return new(K, mixtures, π)
    # end
end
BelugaMixture(K::Int, mixtures, π::Vector{T}, prior) where {T<:Real} =
    return BelugaMixture{T}(K, mixtures, π, prior)

function (model::BelugaMixture)(θ::Vector{V}) where V<:Real
    K = model.K
    Q = getwgdcount(model)
    L = length(model.mixtures[1]) - 2Q
    T = 2L + Q + 1
    m = model.mixtures[1] # temporary model

    # Get the parts of the vector we need
    λ = θ[1:K*L]
    μ = θ[K*L+1:2K*L]
    q = θ[2K*L+1:2K*L+K*Q]
    η = θ[end-2K+1:end-K]

    mixtures = Vector{DLWGD}()
    for i in 1:K
        λ_ = λ[(i-1)*L+1:i*L]
        μ_ = μ[(i-1)*L+1:i*L]
        q_ = Q == 0 ? [] : q[(i-1)*Q+1:i*Q]
        η_ = η[i]
        push!(mixtures, m([λ_..., μ_..., q_..., η_]))
    end

    π = θ[end-K+1:end]
    return BelugaMixture{V}(K, mixtures, π, model.prior)
end

function (model::BelugaMixture)(nt::NamedTuple)
    K = model.K
    Q = getwgdcount(model)
    L = length(model.mixtures[1]) - 2Q
    m = model.mixtures[1] # temporary model

    mixtures = Vector{DLWGD}()
    for i in 1:K
        λ = nt.λ[(i-1)*L+1:i*L]
        μ = nt.μ[(i-1)*L+1:i*L]
        q = Q == 0 ? [] : nt.q[(i-1)*Q+1:i*Q]
        η = nt.η[i]
        push!(mixtures, m([λ..., μ..., q..., η]))
    end
    return BelugaMixture(K, mixtures, nt.π, model.prior)
end

function getwgdcount(model::BelugaMixture)
    return length(Beluga.getwgds(model.mixtures[1]))
end

function params(model::BelugaMixture)
    λ = Float64[]; μ = Float64[]; q = Float64[]; η = Float64[]
    K = model.K
    Q = getwgdcount(model)
    L = length(model.mixtures[1]) - 2Q

    for i in 1:K
        v = asvector(model.mixtures[i])
        append!(λ, v[1:L])
        append!(μ, v[L+1:2L])
        append!(q, v[2L+1:end-1])
        push!(η, v[end])
    end

    if Q == 0
        return (λ=λ, μ=μ, η=η, π=model.π)
    else
        return (λ=λ, μ=μ, q=q, η=η, π=model.π)
    end
end

function PhyloVI.asvector(model::BelugaMixture)
    vec = params(model)
    vec = collect(Iterators.flatten(vec))
    return vec
end

function model_transform(model::BelugaMixture)
    K = model.K
    Q = getwgdcount(model)
    L = length(model.mixtures[1]) - 2Q

    t = nothing
    if Q != 0
        t = as((λ = as(Array, as_positive_real, K*L),
                μ = as(Array, as_positive_real, K*L),
                q = as(Array, as_unit_interval, K*Q),
                η = as(Array, as_unit_interval, K),
                π = UnitSimplex(K)))
    else
        t = as((λ = as(Array, as_positive_real, K*L),
                μ = as(Array, as_positive_real, K*L),
                η = as(Array, as_unit_interval, K),
                π = UnitSimplex(K)))
    end
    return t
end

function model_invtransform(model::BelugaMixture)
    t = inverse(model_transform(model))
    return t
end

function logprior(model::BelugaMixture)
    lp = 0.0
    for i in 1:model.K
        lp += logpdf(model.prior, model.mixtures[i])
    end
    d = Dirichlet(repeat([1/model.K], model.K))
    lp += logpdf(d, model.π)
    return lp
end

# function grad_logprior(model::BelugaMixture)
#     # Wirte the gradient function here...
# end

function loglikelihood(model::BelugaMixture, data::AbstractArray{T}) where T
    ll = 0.0
    for i in 1:length(data)
        temp = 0.0
        for k in 1:model.K
            temp += model.π[k] * exp(logpdf(model.mixtures[k], data[i].xp))
        end
        ll += log(temp)
    end
    return ll
end

# This overloads the existing function
function loglikelihood(model::WGDBelugaModel, data::AbstractArray{T}) where T
    l = zero(Float64)
    @inbounds for i in 1:length(data)
        l += logpdf(model.model, data[i].xp)
    end
    return l
end

# This overloads the existing function
function grad_loglikelihood(model::WGDBelugaModel, data::AbstractArray{T}) where T
    # Get current model parameters
    θ = collect(Iterators.flatten(params(model)))

    # Define helper function for gradient
    helper(x) = begin
        return ll(model(x), data)
    end

    return ForwardDiff.gradient(helper, θ)
end

using CSV
Random.seed!(12)
# Get tree and simulation data
datadir = "./PhyloVI/data/"
tree = open(joinpath(datadir, "species_trees/plants2.nw"), "r") do f ; readline(f); end
df = CSV.read(joinpath(datadir, "branch_wise/1.counts.csv"), delim=",")

# Init model
λ, μ, η = 1., 1., 0.5
model, profile = DLWGD(tree, df, λ, μ, η)

# Simulate some data
rr = CSV.read(".\\PhyloVI\\data\\branch_wise\\1.rates.csv", delim=",")
η = 0.85
x = [rr[:λ]..., rr[:μ]..., η]
model = model(x)
df = rand(model, 1000)
model, profile = DLWGD(tree, df, λ, μ, η)

data = profile
prior = IidRevJumpPrior(
    Σ₀=[0.5 0.45 ; 0.45 0.5],
    X₀=MvNormal(log.(ones(2)), [0.5 0.45 ; 0.45 0.5]),
    πK=DiscreteUniform(0,20),
    πq=Beta(1,1),
    πη=Beta(3,1),
    Tl=treelength(model))

K = 3
m = repeat([model], K)
b = BelugaMixture(K, m, repeat([1/K], K), prior)

D = dimension(model_transform(b))
elbo = ELBO(1)
q = MeanFieldGaussian([zeros(D)..., ones(D)/10...])

# opt = ADAM(0.1)
# advi = ADVI(1, 500, 2, 100, 10^-5, VarInfLogger(DataFrame(repeat([Float64[]], 4*D+1))))
# Q = optimize(advi, elbo, q, b, data, opt)
# res = asvector(Q)[1:D]
# r = model_transform(b)(res)
# println(r)
#
# df = advi.logger.df
# best_index = argmax(df[:, end])
# ζ = collect(df[best_index, :])
# θ = model_transform(b)(ζ[1:D])
# print("Best result: ", θ)
#
# truth = [rr[:λ]..., rr[:μ]..., η]
# function plots(index)
#     p = 0
#     for i in 1:K
#         if i == 1
#             p = StatsPlots.plot(LogNormal(ζ[index+(i-1)*L], exp(ζ[D+index+(i-1)*L])), legend=:none)
#         else
#             StatsPlots.plot!(LogNormal(ζ[index+(i-1)*L], exp(ζ[D+index+(i-1)*L])), legend=:none)
#         end
#     end
#     Plots.vline!([truth[index]])
#     return Plots.plot(p)
# end
# p = [plots(i) for i in [1, 2, 3, 4, 5, 6, 15]]
# Plots.plot(p...)
#
# Plots.plot(df[:, end])




# Try some stuff here with BBVI
ζ = rand(q)
logp(b, ζ, data)
logq(q, ζ)
∇logq(q, ζ)

opt = ADAM(0.1)
Q, hist = optimize(200, 10, q, b, opt, data)
θ = Distributions.params(Q)[:μ]
r = model_transform(b)(θ)
R = asvector(Q)
truth = [rr[:λ]..., rr[:μ]..., η]

using Measures
function pl(index)
    p = 0
    for i in 1:K
        if i == 1
            p = StatsPlots.plot(LogNormal((R[index+(i-1)*L]), (R[D+index+(i-1)*L])), legend=:none, tickfont=font(4))
        else
            StatsPlots.plot!(LogNormal((R[index+(i-1)*L]), (R[D+index+(i-1)*L])), legend=:none, tickfont=font(4))
        end
        Plots.vline!([truth[index]])
    end
    return Plots.plot(p, margins=0mm)
end
p = [pl(i) for i in 1:L]
Plots.plot(p..., fmt=:svg, margins=-10mm)
