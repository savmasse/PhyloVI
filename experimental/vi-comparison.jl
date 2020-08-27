using Plots
using StatsPlots
using Flux.Optimise
using Measurements

# Create data from a Mixture model
true_gmm = GaussianMixtureModel(2, [-1., 1.], [0.3, 0.5], [0.3, 0.7])
data = rand(true_gmm, 1000)

p = Plots.histogram(data, bins=100,
                    normalize=true,
                    alpha=0.3,
                    label="data", fmt=:svg)
x = collect(-2:0.01:3)
Plots.plot!(x, 0.7 .* pdf.(Normal(1, 0.5), x) .+ 0.3 .* pdf.(Normal(-1, 0.3), x),
            label="parametric", color="black")
Plots.pdf(p, "vi_comp_data")

# Setup advi
gmm = GaussianMixtureModel(2, [-3., 3.], rand(2), [0.5, 0.5])
q = MeanFieldGaussian([zeros(5)..., (ones(5))...])
logger = VarInfLogger(DataFrame(repeat([Float64[]], 21)))
elbo = ELBO(1)
advi = ADVI(1, 200, 2, 100, 0.01, logger)
opt = ADAM(0.1)

# Perform optimization
Q_advi = optimize(advi, elbo, q, gmm, data, opt)

# Analyse results
res_advi = model_transform(gmm)(Distributions.params(Q_advi).μ)


# Now set up BBVI for same model
opt = ADAM(0.1)
Q_bbvi, y = optimize(500, 20, q, gmm, opt, data, 200)
res_bbvi = model_transform(gmm)(Distributions.params(Q_bbvi).μ)


# Same but for SVGD
opt = ADAM(0.1)
M = 100
x0 = rand(M, 5)
svgd = SVGD(M, 100, opt, -1, VarInfLogger())
X_svgd = optimize(svgd, gmm, data, 1000, x0)


α = 0.6
p = []
# Comparison plots of the different methods
push!(p, StatsPlots.plot(Normal(Q_advi.dists[1].μ, Q_advi.dists[1].σ), label="ADVI", fill=true,
        alpha=α, normalize=true, margin=0mm, legend=:topright, legendfontsize=3,
        tickfont=font(6), bottom_margin=2mm, xlabel="\\(a\\) \\mu_1"))
StatsPlots.plot!(Normal(Q_bbvi.dists[1].μ, Q_bbvi.dists[1].σ), label="BBVI", fill=true, alpha=α, normalize=true)
Plots.histogram!(X_svgd[:, 1], bins=Int(2M), normalize=false, label="SVGD", color=:black)
Plots.vline!([-1.], color=:red, label="true value", linestyle=:dot, linewidth=2)

push!(p, StatsPlots.plot(Normal(Q_advi.dists[2].μ, Q_advi.dists[2].σ), label="ADVI", fill=true,
        alpha=α, normalize=true, margin=0mm, legend=:topleft, legendfontsize=3,
        tickfont=font(6), bottom_margin=2mm, xlabel="\\(b\\) \\mu_2"))
StatsPlots.plot!(Normal(Q_bbvi.dists[2].μ, Q_bbvi.dists[2].σ), label="BBVI", fill=true, alpha=α, normalize=true)
Plots.histogram!(X_svgd[:, 2], bins=Int(2M), normalize=false, label="SVGD", color=:black)
Plots.vline!([1.], color=:red, label="true value", linestyle=:dot, linewidth=2)

push!(p, StatsPlots.plot(LogNormal(Q_advi.dists[3].μ, Q_advi.dists[3].σ), label="ADVI", fill=true,
        alpha=α, normalize=true, margin=0mm, legend=:topleft, legendfontsize=3,
        tickfont=font(6), xlabel="\\(c\\) \\sigma_1"))
StatsPlots.plot!(LogNormal(Q_bbvi.dists[3].μ, Q_bbvi.dists[3].σ), label="BBVI", fill=true, alpha=α, normalize=true)
Plots.histogram!(exp.(X_svgd[:, 3]), bins=Int(2M), normalize=false, label="SVGD", color=:black)
Plots.vline!([0.3], color=:red, label="true value", linestyle=:dot, linewidth=2)

push!(p, StatsPlots.plot(LogNormal(Q_advi.dists[4].μ, Q_advi.dists[4].σ), label="ADVI", fill=true,
        alpha=α, margin=0mm, legend=:topleft, legendfontsize=3,
        tickfont=font(6), xlabel="\\(d\\) \\sigma_2"))
StatsPlots.plot!(LogNormal(Q_bbvi.dists[4].μ, Q_bbvi.dists[4].σ), label="BBVI", fill=true, alpha=α)
Plots.histogram!(exp.(X_svgd[:, 4]), bins=Int(2M), normalize=false, label="SVGD", color=:black)
Plots.vline!([0.5], color=:red, label="true value", linestyle=:dot, linewidth=2)

x = Plots.plot(p...)
Plots.pdf(x, "vi-comp")


# Compare variance on the gradient estimator
S = 100

a = []
b = []
c = []
X = [1, 10, 100, 1_000, 10_000]
for N in X
    bbvi_est = [bbvi_estimator(N, q, gmm, data)[1] for _ in 1:S]
    bbvi_cv_est = [bbvi_estimator_cv(N, q, gmm, data)[1] for _ in 1:S]
    advi_est = [calc_grad_elbo(q, gmm, data, N)[1] for _ in 1:S]

    println("BBVI estimator variance (n=", N, ") : ", var(bbvi_est))
    println("BBVI-CV estimator variance (n=", N, ") : ", var(bbvi_cv_est))
    println("ADVI estimator variance (n=", N, ") : ", var(advi_est))
    println()

    append!(a, var(bbvi_est))
    append!(b, var(bbvi_cv_est))
    append!(c, var(advi_est))
end

p = Plots.plot(X, a, yscale=:log, xscale=:log, label="BBVI", xlabel="Number of MC samples", ylabel="Variance")
Plots.plot!(X, b, yscale=:log, xscale=:log, label="BBVI-CV")
Plots.plot!(X, c, yscale=:log, xscale=:log, label="ADVI")
Plots.pdf(p, "variance_plot")
