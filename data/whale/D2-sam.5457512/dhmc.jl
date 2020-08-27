@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using Whale
using Whale.NewickTree, Distributions, DynamicHMC, DataFrames, LinearAlgebra, Random, CSV
using DynamicHMC.Diagnostics

treefile = string(@__DIR__, "/D2.nw") #ARGS[1]
alefile  = string(@__DIR__, "/ale/") #ARGS[2]
outdir   = @__DIR__ #ARGS[3]
eta      = 0.67
niter    = 2

tree = readnw(readline(treefile))
# Add all WGDs on internal branches (~ MAPS)
n = length(prewalk(tree))
nwgd = 5
for n in prewalk(tree)
    (isroot(n) || isleaf(n)) && continue
    insertnode!(n, name="wgd_$(id(n))")
end

rates = RatesModel(Whale.DLWGD(λ=randn(n), μ=randn(n), q=rand(nwgd), η=rand()), fixed=(:p,))
model = WhaleModel(rates, tree, 0.01)
data  = read_ale(alefile, model, true)

prior   = Whale.IWIRPrior(Ψ=[1. 0. ; 0. 1.], πr=MvNormal([1., 1.]))
problem = WhaleProblem(data, model, prior)

# results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, niter,
#     warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=2))
# @info summarize_tree_statistics(results.tree_statistics)
#
# posterior = Whale.transform(problem, results.chain)
# df = Whale.unpack(posterior)

#
# path = mkpath(outdir)
# CSV.write(joinpath(path, "dhmc-$niter.csv"), df)


# Create a WGD model
m = WGDWhaleModel(prior, rates, tree, 0.01)
D = dimension(model_transform(m))

# Test ADVI
opt = ADAM(.02)
x = [Normal(0, 0.05) for _ in 1:D-6]  # Starting values for λ, μ
append!(x, [Normal(-2, 0.05) for _ in 1:5])  # Starting values for q
push!(x, Normal(0.1, 0.05))  # Starting value for η
q = MeanFieldGaussian(x)
elbo = ELBO(1)
logger = VarInfLogger(DataFrame(repeat([Float64[]], 4D+1)))
advi = ADVI(1, 200, 2, 100, 10^-3, logger)
Q = optimize(advi, elbo, q, m, data, opt)


# Do same with BBVI
Random.seed!(123)
x = [Normal(0, 0.1) for _ in 1:D-6]
append!(x, [Normal(-2, 0.05) for _ in 1:5])  # Starting values for q
push!(x, Normal(1., 0.1))
q = MeanFieldGaussian(x)
opt = ADAM(0.01)
logger_bbvi = VarInfLogger(DataFrame(repeat([Float64[]], 4D+1)))
@time Q_bbvi = optimize(1000, 20, q, m, opt, data, 100, logger_bbvi)
res_bbvi = Distributions.params(Q_bbvi).μ
μ_bbvi = model_transform(m)(res_bbvi)
σ_bbvi = Distributions.params(Q_bbvi).σ
ζ_bbvi = collect(logger_bbvi.df[end, :])


# Make some plots
df_mcmc = CSV.read(string(@__DIR__, "/dhmc-1000.csv"))

p = [collect(Iterators.flatten(model_transform(m, i))) for i in rand(Q_bbvi, 5_000)]
pa = [collect(Iterators.flatten(model_transform(m, i))) for i in rand(Q, 5_000)]

df = DataFrame(transpose(hcat(p...)))
dfa = DataFrame(transpose(hcat(pa...)))
p1 = @df df_mcmc StatsPlots.violin([:q_1, :q_2, :q_3, :q_4, :q_5],
                                marker=(0.2,:blue,stroke(0)),
                                xticks=1:5, legend=:none, color="blue", alpha=0.8, ylabel="q")
StatsPlots.violin!([[i] for i in 1:5], [df[:x27], df[:x28], df[:x29], df[:x30], df[:x31]],
                                marker=(0.2,:blue,stroke(0)),
                                xticks=1:5, legend=:none, color="red", alpha=0.8)
StatsPlots.violin!([[i] for i in 1:5], [dfa[:x27], dfa[:x28], dfa[:x29], dfa[:x30], dfa[:x31]],
                                marker=(0.2,:blue,stroke(0)),
                                xticks=1:5, legend=:none, color="orange", alpha=0.8)
pl = Plots.plot(p1, fmt=:svg)
# Plots.pdf(pl, "whale-q")




p = []
L = 13
Q_advi = Q

for index in 1:L
    # Create empty plot
    push!(p, Plots.plot(legendfontsize=4,
                        tickfont=font(4),
                        legend=:none,
                        title=string("\\lambda_", index),
                        titlefont=font(7),
                        margin=-4mm, fmt=:svg))

    # Add BBVI
    StatsPlots.plot!(LogNormal((Q_bbvi.dists[index].μ),
                            Q_bbvi.dists[index].σ),
                            fill=true,
                            label="BBVI",
                            alpha=0.95)

    # Add ADVI
    StatsPlots.plot!(LogNormal((Q_advi.dists[index].μ),
                Q_advi.dists[index].σ),
                fill=true,
                label="ADVI",
                alpha=0.8)

    # Add MCMC with 1000 sample burn-in
    Plots.histogram!(exp.(df_mcmc[Symbol(string("λ_", index))]),
                    label="MCMC",
                    normalize=true,
                    bins=30,
                    alpha=0.5)
end
# legend = plot([0 0 0 0], showaxis = false, grid = false, label = ["BBVI" "ADVI" "true value" "MCMC"])
legend = plot([0 0 0], showaxis = false, grid = false, label = ["BBVI" "ADVI" "MCMC"])
Plots.plot(p..., legend, size=(800, 550))
Plots.pdf("whale-distribution-lambda")

p = []
for index in L+1:2L
    # Create empty plot
    push!(p, Plots.plot(legendfontsize=4,
                        tickfont=font(4),
                        legend=:none,
                        title=string("\\mu_", index-L),
                        titlefont=font(7),
                        margin=-5mm, fmt=:svg))

    # Add BBVI
    StatsPlots.plot!(LogNormal((Q_bbvi.dists[index].μ),
                            Q_bbvi.dists[index].σ),
                            fill=true,
                            label="BBVI",
                            alpha=0.95)

    # Add ADVI
    StatsPlots.plot!(LogNormal((Q_advi.dists[index].μ),
                Q_advi.dists[index].σ),
                fill=true,
                label="ADVI",
                alpha=0.8)

    # Add MCMC with 1000 sample burn-in
    Plots.histogram!(exp.(df_mcmc[Symbol(string("μ_", index-L))]),
                    label="MCMC",
                    normalize=true,
                    bins=30,
                    alpha=0.5)
end
# legend = plot([0 0 0 0], showaxis = false, grid = false, label = ["BBVI" "ADVI" "true value" "MCMC"])
legend = plot([0 0 0], showaxis = false, grid = false, label = ["BBVI" "ADVI" "MCMC"])
Plots.plot(p..., legend, size=(800, 550))
Plots.pdf("whale-distribution-mu")
