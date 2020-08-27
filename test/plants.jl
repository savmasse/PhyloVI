
using Beluga, CSV, DataFrames, Parameters
using Measures, IterTools, BenchmarkTools
using PhyloVI
using Flux.Optimise

include("../src/vimodel.jl")
include("../src/phylomodel.jl")
include("../src/meanfield.jl")
include("../src/elbo.jl")
include("../src/logger.jl")
include("../src/advi.jl")
include("../src/wgdbelugamodel.jl")

# Seed for reproducibility
Random.seed!(1)
ITERATIONS = 300

# get some data
datadir = "./PhyloVI/data/"
tree = open(joinpath(datadir, "species_trees/plants2.nw"), "r") do f ; readline(f); end
# df = CSV.read(joinpath(datadir, "real_world/plants1.filter012.csv"), delim=",")
df = CSV.read(joinpath(datadir, "branch_wise/1.counts.csv"), delim=",")


# Simulate some more data
rr = CSV.read(".\\PhyloVI\\data\\branch_wise\\1.rates.csv", delim=",")
η = 0.85
λ, μ = 1., 1.
model, profile = Beluga.DLWGD(tree, df, λ, μ, η)

# insertwgd!(model, model[6], 0.05, 0.25)
# extend!(profile, 6);

data = profile
prior = IidRevJumpPrior(
    Σ₀=[0.5 0.45 ; 0.45 0.5],
    X₀=MvNormal(log.(ones(2)), [0.5 0.45 ; 0.45 0.5]),
    πK=DiscreteUniform(0,20),
    πq=Beta(1,1),
    πη=Beta(3,1),
    Tl=treelength(model))

# Create a Beluga model with a WGD
bm = WGDBelugaModel(prior, model)
N = length(Beluga.getwgds(model))
L = length(model) - 2N
T = 2*L + N + 1         # (L x λ, L x μ, N x q, and 1 x η)
x = [Normal(log(.5), 0.01) for _ in 1:T]
# push!(x, Normal(log(1.), 0.05))
q = MeanFieldGaussian(x)
elbo = ELBO(1)


################################################################################
#                                   ADVI                                       #
################################################################################
Random.seed!(1)
opt = ADAM(0.01)
x_advi = [Normal(log(.5), 0.1) for _ in 1:T]
q_advi = MeanFieldGaussian(x_advi)
advi = ADVI(1, 2000, 2, 100, 1e-4, VarInfLogger(N, L))
@time Q_advi = optimize(advi, elbo, q_advi, bm, data, opt)

# Compare with simulated values
r = Distributions.params(Q_advi)[:μ]
res = model_transform(bm)(r)
d = DataFrame((truth=rr[:λ], vi=res[:λ]))


################################################################################
#                                   BBVI                                       #
################################################################################

Random.seed!(1)
logger_bbvi = VarInfLogger(N, L)
x_bbvi = [Normal(log(.5), 0.1) for _ in 1:T]
q_bbvi = MeanFieldGaussian(x_bbvi)
opt = ADAM(0.01)
@time Q_bbvi = optimize(2000, 20, q_bbvi, bm, opt, data, 100, logger_bbvi)


################################################################################
#                                   SVGD                                       #
################################################################################

# Random.seed!(1)
# opt = ADAM(0.01)
# logger_svgd = VarInfLogger(N, L)
# svgd = SVGD(1, 1, opt, -1, logger_svgd)
# @time X = optimize(svgd, bm, data, 200)

Random.seed!(1)
opt = ADAM(0.04)
logger_svgd50 = VarInfLogger(N, L)
svgd50 = SVGD(50, 200, opt, -1, logger_svgd50)
@time X50 = optimize(svgd50, bm, data, 20)


################################################################################
#                                   MCMC                                       #
################################################################################
Random.seed!(1)
chain = RevJumpChain(data=data, model=deepcopy(model), prior=prior)
init!(chain)
@time rjmcmc!(chain, 3000, show=10, trace=1)

# Save MCMC values to csv
CSV.write("./results/simulation_results_mcmc.csv", chain.trace)

# Plot the logs and compare to the true values
# plot_logs(advi.logger, q, bm, 4, L, N, T, false)
# plot_logs(advi.logger, q, bm, 4, L, N, T, true)

################################################################################
#                                 Analysis                                     #
################################################################################
using Measurements

plogger_advi = to_param_space(advi.logger, q, bm)
plogger_bbvi = to_param_space(logger_bbvi, q, bm)
# plogger_svgd = to_param_space(logger_svgd, q, bm)

p = []
parnames = Dict( Symbol(string("λ", i)) => string("\\lambda_", i) for i in 1:L )
λ_truth = Dict( Symbol(string("λ", i)) => rr[:λ][i] for i in 1:L )
for s in keys(parnames)
    push!(p, Plots.plot(legendfontsize=4,
                        tickfont=font(4),
                        legend=:none,
                        title=get(parnames, s, nothing);
                        titlefont=font(7),
                        margin=-1mm))
    Plots.plot!(plogger_advi[s], label="ADVI")
    Plots.plot!(plogger_bbvi[s], label="BBVI")
    # Plots.plot!(plogger_svgd[s], label="SVGD (MAP)")
    Plots.hline!([λ_truth[s]])
end
legend = plot([0 0 0], showaxis = false, grid = false, label = ["ADVI" "BBVI" "true value"])
Plots.plot(p..., legend)
Plots.pdf("./images/convergence-lambda")

p = []
parnames = Dict( Symbol(string("μ", i)) => string("\\mu_", i) for i in 1:L )
μ_truth = Dict( Symbol(string("μ", i)) => rr[:μ][i] for i in 1:L )
for i in 1:L
    s = Symbol(string("μ", i))
    push!(p, Plots.plot(legendfontsize=4,
                        tickfont=font(4),
                        legend=:none,
                        title=get(parnames, s, nothing);
                        titlefont=font(7),
                        margin=-1mm, topmargin=-10mm))
    Plots.plot!(plogger_advi[s], label="ADVI")
    Plots.plot!(plogger_bbvi[s], label="BBVI")
    # Plots.plot!(plogger_svgd[s], label="SVGD (MAP)")
    Plots.hline!([μ_truth[s]])
end
legend = plot([0 0 0], showaxis = false, grid = false, label = ["ADVI" "BBVI" "true value"])
Plots.plot(p..., legend)
Plots.pdf("./images/convergence-mu")



# Score comparison between ADVI and BBVI
Plots.plot(advi.logger.df[end], label="ADVI", title="Score", fmt=:svg,
            xlabel="# iterations", margin=-2mm)
Plots.plot!(logger_bbvi.df[end], label="BBVI")
Plots.plot!((chain.trace[:logp] .+ chain.trace[:logπ]), label="MCMC")
Plots.pdf("beluga-sim-score")


p = []
for index in 1:L
    # Create empty plot
    push!(p, Plots.plot(legendfontsize=4,
                        tickfont=font(4),
                        legend=:none,
                        title=string("\\lambda_", index),
                        titlefont=font(7),
                        margin=-4mm, fmt=:svg))

    # # Add BBVI
    # StatsPlots.plot!(LogNormal((Q_bbvi.dists[index].μ),
    #                         Q_bbvi.dists[index].σ),
    #                         fill=true,
    #                         label="BBVI",
    #                         alpha=0.95)
    # Add SVGD
    Plots.histogram!(exp.(X50[:, index]),
                    label="SVGD",
                    alpha=0.7,
                    normalize=true;
                    bins=20)
    # # Add ADVI
    # StatsPlots.plot!(LogNormal((Q_advi.dists[index].μ),
    #             Q_advi.dists[index].σ),
    #             fill=true,
    #             label="ADVI",
    #             alpha=0.8)
    # Add ground truth values
    Plots.vline!([rr.λ[index]],
                label="True value",
                # linestyle=:dot,
                linewidth=3)
    # Add MCMC with 1000 sample burn-in
    Plots.histogram!(chain.trace[Symbol(string("λ", index))][1:end],
                    label="MCMC",
                    normalize=true,
                    bins=30,
                    alpha=0.5)
end
legend = plot([0 0 0], showaxis = false, grid = false, label = ["SVGD" "true value" "MCMC"])
Plots.plot(p..., legend, size=(800, 550))
Plots.pdf("./images/beluga-sim-distribution-mu")


p = []
for index in L+1:2L
    # Create empty plot
    push!(p, Plots.plot(legendfontsize=4,
                        tickfont=font(4),
                        legend=:none,
                        title=string("\\mu_", index-L),
                        titlefont=font(7),
                        margin=-4mm, fmt=:svg))

    # # Add BBVI
    # StatsPlots.plot!(LogNormal((Q_bbvi.dists[index].μ),
    #                         Q_bbvi.dists[index].σ),
    #                         fill=true,
    #                         label="BBVI",
    #                         alpha=0.95)
    # Add SVGD
    Plots.histogram!(exp.(X50[:, index]),
                    label="SVGD",
                    alpha=0.7,
                    normalize=true;
                    bins=20)
    # # Add ADVI
    # StatsPlots.plot!(LogNormal((Q_advi.dists[index].μ),
    #             Q_advi.dists[index].σ),
    #             fill=true,
    #             label="ADVI",
    #             alpha=0.8)
    # Add ground truth values
    Plots.vline!([rr.μ[index-L]],
                label="True value",
                # linestyle=:dot,
                linewidth=2)
    # Add MCMC with 1000 sample burn-in
    Plots.histogram!(chain.trace[Symbol(string("μ", index-L))][1000:end],
                    label="MCMC",
                    normalize=true,
                    bins=30,
                    alpha=0.5)
end
# legend = plot([0 0 0 0], showaxis = false, grid = false, label = ["BBVI" "ADVI" "true value" "MCMC"])
legend = plot([0 0 0], showaxis = false, grid = false, label = ["SVGD" "MCMC" "true value"])
Plots.plot(p..., legend, size=(800, 550))
# Plots.pdf("beluga-sim-distribution-mu")
