################################################################################
######      Test new ADVI functionality         ################################
################################################################################

function optimize(advi::ADVI,
                  elbo::ELBO,
                  q::M,
                  model::D,
                  data::AbstractArray,
                  opt) where {M<:MeanField, D<:PhyloModel}

    # Transform variational params to real-space
    ζ = sample_invtransform(q, Distributions.params(q))
    Q = q
    Q_best = q
    score_best = -Inf
    elbo_list = Vector{Float64}()

    # Show user the batching situation
    if advi.verbose > 1
        println("Using minibatches of size ", advi.batch_size, ".")
    end

    # Iterate until convergence or maximum of iterations is reached.
    for i in 1:advi.max_iter

        # Sample minibatch and convert back to distributed object
        batch = StatsBase.sample(data, advi.batch_size, replace=false)
        batch = distribute(collect(batch))
        batch_factor = size(data)[1] / advi.batch_size

        # Calculate gradient update
        ∇ζ = calc_grad_elbo(Q, model, batch, advi.n_samples, batch_factor)
        Optimise.update!(opt, ζ, -∇ζ)

        # Create updated variational distribution
        θ = sample_transform(Q, ζ)
        Q = M(θ)

        # Calculate the ELBO objective
        curr_elbo = elbo(Q, model, data)
        avg_elbo = rolling_average!(elbo_list, curr_elbo)
        if advi.verbose > 1; println("Iteration ", i, ": ELBO=", curr_elbo, " (", avg_elbo, ")"); end

        if curr_elbo > score_best
            Q_best = Q
            score_best = curr_elbo
        end

        # Log the results
        update_logger!(advi.logger, ζ, ∇ζ, avg_elbo)
    end

    return Q_best
end

# function grad_logprior(model::GaussianMixtureModel)
#     θ = params(model)
#     ζ = model_invtransform(model)(θ)
#
#     helper(x) = begin
#         θ = collect(Iterators.flatten(model_transform(model)(x)))
#         return logprior(model(θ))
#     end
#
#     return ForwardDiff.gradient(helper, ζ)
# end
#
# function grad_loglikelihood(model::GaussianMixtureModel, data::AbstractArray{T}) where T<:Real
#     θ = params(model)
#     ζ = model_invtransform(model)(θ)
#
#     helper(x) = begin
#         θ = collect(Iterators.flatten(model_transform(model)(x)))
#         return loglikelihood(model(θ), data)
#     end
#
#     return ForwardDiff.gradient(helper, ζ)
# end

# # Generate some easy GMM data
# Random.seed!(1)
# K = 2
# μ = [-5., 5.]
# σ = [1., 1.]
# π = [0.3, 0.7]
# θ = [μ, σ, π]
# true_model = GaussianMixtureModel(K, μ, σ, π)
# data = rand(true_model, 1000)
# model = GaussianMixtureModel(K)
#
# d = dimension(model_transform(model))
# μ = zeros(d)
# σ = ones(d)/100
# q = MeanFieldGaussian([μ..., σ...])
# q = FullRankGaussian(MvNormal(μ, σ))
# D = dimension(sample_transform(q))
#
# opt = RMSProp(0.1)
# advi = ADVI(1, 500, 2, 100, 10^-5, VarInfLogger(DataFrame(repeat([Float64[]], 2D+1))))
# Q = optimize(advi, elbo, q, model, data, opt)
# res = model_transform(model)(Distributions.params(Q)[:μ])
# println("Result: ", res)
#
# df = advi.logger.df
# best_index = argmax(df[:, end])
# ζ = collect(df[best_index, :])
# θ = model_transform(model)(ζ[1:d])
# print("Best result: ", θ)
#
# Plots.plot(advi.logger.df[:, end])
