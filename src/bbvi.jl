
# Run optimization process
function optimize(max_iter::Int, N::Int, q::M, model, opt, data, batch_size, logger) where M<:MeanField

    Q = q
    λ = sample_invtransform(q)(Distributions.params(q))
    hist = Vector{Float64}()
    best = [-Inf, q]
    elbo = ELBO(1)
    elbo_list = Float64[]

    # Subsample a small dataset for evaluation purposes
    # holdout = StatsBase.sample(data, Int(size(data)[1]/10), replace=false)
    # holdout = distribute(collect(holdout))
    # train_data = distribute(setdiff(collect(data), holdout))
    train_data = data

    try
        for i in 1:max_iter

            # Sample minibatch and convert back to distributed object
            batch = StatsBase.sample(train_data, batch_size, replace=false)
            batch = distribute(collect(batch))
            batch_factor = size(train_data)[1] / batch_size

            ∇λ = bbvi_estimator_cv(N, Q, model, batch, batch_factor)
            Optimise.update!(opt, λ, -∇λ)

            θ = sample_transform(q)(λ)
            Q = M(θ)

            # curr_elbo = elbo(Q, model, holdout, 10.)
            curr_elbo = elbo(Q, model, data)
            avg_elbo = rolling_average!(elbo_list, curr_elbo)
            println("Iteration ", i, ": ELBO=", curr_elbo, " (", avg_elbo, ")")

            if curr_elbo >= best[1]
                best[2] = Q
                best[1] = curr_elbo
            end

            update_logger!(logger, λ, ∇λ, avg_elbo)
        end
    catch e
        # If error return the last best Q; otherwise we will have no result at all
        # and computation time was wasted.
        println("An error occurred in the optimisation process. The last best Q has been returned.")
        println(e)
        return best[2]

    end

    return best[2]
end

function bbvi_estimator(N::Int, q, model, data, batch_factor=1.0)

    ∇λ = zeros(dimension(sample_transform(q)))

    for i in 1:N
        ζ = rand(q)
        ∇λ += ∇logq(q, ζ) .* (logp(model, ζ, data, batch_factor) - logq(q, ζ))
    end

    return ∇λ ./ N
end

function bbvi_estimator_cv(N::Int, q, model, data, batch_factor=1.0)

    D = dimension(sample_transform(q))
    ∇λ = zeros(D)
    f = zeros(D, N)
    h = zeros(D, N)

    for i in 1:N
        ζ = rand(q)
        lq = ∇logq(q, ζ)
        f[:, i] = lq .* (logp(model, ζ, data, batch_factor) - logq(q, ζ))
        h[:, i] = lq
    end

    # Now calculate the covariance factor a
    if N > 1
        d = [cov(f[i, :], h[i, :]) for i in 1:D]
        n = [std(h[i, :])^2 for i in 1:D]
        a = sum(d) / sum(n)

        ∇λ = sum(f .- a .* h, dims=2)
        ∇λ = vec(∇λ)
    else
        ∇λ = f
        ∇λ = vec(∇λ)
    end

    return ∇λ ./ N
end

# function obbvi_estimator(N::Int, q::MeanFieldGaussian, model, data, τ)
#
#     # Create a new overdispersed q
#     θ = asvector(q)
#     θ[length(q):end] .*= τ
#     r = MeanFieldGaussian(θ)
#
#     D = dimension(sample_transform(q))
#     ∇λ = zeros(D)
#     f = zeros(D, N)
#     h = zeros(D, N)
#
#     function logQ(q::MeanFieldGaussian, r, ζ)
#         weights = zeros(2*length(q))
#
#         for i in 1:length(q)
#             pμ = pdf(r.dists[i], r.dists[i].μ) / pdf(q.dists[i], q.dists[i].μ)
#             w = pμ * pdf(q.dists[i], ζ[i]) / pdf(r.dists[i], ζ[i])
#             weights[(i-1)*2 + 1] = w
#             weights[i*2] = w
#         end
#         return weights
#     end
#
#     for i in 1:N
#         ζ = rand(r)
#         # w = τ * exp(logq(q, ζ)) / exp(logq(r, ζ))
#         w = logQ(q, r, ζ)
#         lq = ∇logq(q, ζ)
#         f[:, i] = w .* lq .* (logp(model, ζ, data) - logq(q, ζ))
#         h[:, i] = w .* lq
#     end
#
#     # Now calculate the covariance factor a
#     d = [cov(f[i, :], h[i, :]) for i in 1:D]
#     n = [std(h[i, :])^2 for i in 1:D]
#     a = sum(d) / sum(n)
#
#     ∇λ = sum(f .- a .* h, dims=2)
#     ∇λ = vec(∇λ)
#
#     return ∇λ ./ N
# end

function logp(model::T, ζ, data, batch_factor) where T<:PhyloModel
    t = model_transform(model)
    θ = t(ζ)
    m = model(θ)
    return logprior(m) + loglikelihood(m, data) * batch_factor + TransformVariables.transform_and_logjac(t, ζ)[2]
end

function logq(q::M, ζ) where {T<:PhyloModel, M<:MeanField}
    r = 0.0
    for i in 1:length(q)
        r += logpdf(q.dists[i], ζ[i])
    end
    return r
end

# OK to do AD here because computation is negligable compared to logp
function ∇logq(q::M, ζ) where M<:MeanField
    # λ are the variational parameters
    λ = sample_invtransform(q)(Distributions.params(q))
    function helper(λ)
        θ = sample_transform(q)(λ)
        Q = M(θ)
        return logq(Q, ζ)
    end
    return ForwardDiff.gradient(helper, λ)
end

# using Flux.Optimise

# Random.seed!(1)
# # K = 3
# # μ = [-5., 0., 5.]
# # σ = [0.5, 0.8, 1.]
# # π = [0.3, 0.5, 0.2]
# K = 2
# μ = [-5., 5.]
# σ = [0.5, 0.8]
# π = [0.3, 0.7]
# θ = [μ, σ, π]
# true_model = GaussianMixtureModel(K, μ, σ, π)
# data = rand(true_model, 500)
#
# # Test the Guassian Mixture model
# model = GaussianMixtureModel(K, zeros(K), ones(K)/10, repeat([1/K], K))
# D = dimension(model_transform(model))
# q = MeanFieldGaussian([zeros(D)..., ones(D)/100...])
#
# Q, hist = optimize(1000, 3, q, model, ADAM(0.3), data, 500)
# res = Distributions.params(Q)[:μ]
# θ = model_transform(model)(res)
#
# println(θ)
# Plots.plot(hist, legend=:none)

# using CSV
# Random.seed!(12)
# # Get tree and simulation data
# datadir = "./PhyloVI/data/"
# tree = open(joinpath(datadir, "species_trees/plants2.nw"), "r") do f ; readline(f); end
# df = CSV.read(joinpath(datadir, "branch_wise/1.counts.csv"), delim=",")
#
# # Init model
# λ, μ, η = 1., 1., 0.5
# model, profile = DLWGD(tree, df, λ, μ, η)
#
# # Simulate some data
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
# t = model_transform(bm)
# truth = [rr[:λ]..., rr[:μ]..., η]
#
# # Create a suitable initial distribution
# elbo = ELBO(1)
# θ₀ = [zeros(T)..., ones(T)/100...]
# q = MeanFieldGaussian(θ₀)
#
# opt = RMSProp(0.1)
# Q, hist = optimize(260, 10, q, bm, opt, data, 200)
# res = Distributions.params(Q)[:μ]
# θ = model_transform(bm)(res)
#
# println(θ)
# Plots.plot(hist)
