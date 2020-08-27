
using Beluga, CSV, DataFrames, Parameters, BenchmarkTools
using PhyloVI
using Flux.Optimise
using Measurements

include("../src/vimodel.jl")
include("../src/phylomodel.jl")
include("../src/meanfield.jl")
include("../src/elbo.jl")
include("../src/logger.jl")
include("../src/advi.jl")
include("../src/wgdbelugamodel.jl")

Random.seed!(1)

# get some data
datadir = "./PhyloVI/data/"
tree = open(joinpath(datadir, "rice/species_tree.nwk"), "r") do f ; readline(f); end
df = CSV.read(joinpath(datadir, "rice/og-filter012.counts.csv"), delim=",")

# Subsample
indices = rand(1:size(df)[1], 500)
# indices = rand(1:size(df)[1], 5_000)
df = df[indices, :]

# Initiate model
λ, μ, η = 0.5, 0.5, 0.5
model, profile = DLWGD(tree, df, λ, μ, η)

# Insert a WGD node into the tree
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

# Set up initial values of the variational distribution
x = [Normal(-2., 0.05) for _ in 1:T]
# push!(x, Normal(log(0.05), 0.01))
q = MeanFieldGaussian(x)

# Run VI algorithm
elbo = ELBO(1)
advi = ADVI(1, 300, 2, 200, 1e-3, VarInfLogger(N, L))
opt = ADAM(0.02)
Q_advi = optimize(advi, elbo, q, bm, data, opt)
r = Distributions.params(Q_advi)[:μ]
s = Distributions.params(Q_advi)[:σ]
res = model_transform(bm)(r)

# Save the results
CSV.write("./results/rice_results_vi.csv", advi.logger.df)

d = dimension(model_transform(bm))
df = advi.logger.df
best_index = argmax(df[:, end])
ζ = collect(df[best_index, :])
θ = model_transform(bm)(ζ[1:d])
print("Best result: ", θ)


Plots.plot(advi.logger.df[:, T])
# Plots.plot(ch.trace[:η1])


# Do same with BBVI
Random.seed!(123)
opt = ADAM(0.02)
logger_bbvi = VarInfLogger(N, L)
@time Q_bbvi = optimize(1000, 20, q, bm, opt, data, 200, logger_bbvi)
res_bbvi = Distributions.params(Q_bbvi).μ
μ_bbvi = model_transform(bm)(res_bbvi)
σ_bbvi = Distributions.params(Q_bbvi).σ
ζ_bbvi = collect(logger_bbvi.df[end, :])


# Now run the MCMC equivalent on the same data
Random.seed!(123)
ch = RevJumpChain(data=data, model=deepcopy(model), prior=prior)
init!(ch)
@time rjmcmc!(ch, 1000, show=10, trace=1)

CSV.write("./results/rice-results-mcmc.csv", ch.trace)



################################################################################
#                                 Analysis                                     #
################################################################################
using Measurements

p = []
for index in 1:L
    # Create empty plot
    push!(p, Plots.plot(legendfontsize=4,
                        tickfont=font(4),
                        legend=:none,
                        title=string("\\lambda_", index),
                        titlefont=font(7),
                        margin=-3mm, fmt=:svg))

    # Add BBVI
    StatsPlots.plot!(LogNormal((Q_bbvi.dists[index].μ),
                            Q_bbvi.dists[index].σ),
                            fill=true,
                            label="BBVI",
                            alpha=0.95)
    # # Add SVGD
    # Plots.histogram!(X50[:, index],
    #                 label="SVGD",
    #                 alpha=0.7,
    #                 normalize=true;
    #                 bins=10)
    # Add ADVI
    StatsPlots.plot!(LogNormal((Q_advi.dists[index].μ),
                Q_advi.dists[index].σ),
                fill=true,
                label="ADVI",
                alpha=0.8)
    # Add MCMC with 1000 sample burn-in
    Plots.histogram!(ch.trace[Symbol(string("λ", index))][1:end],
                    label="MCMC",
                    normalize=true,
                    bins=20,
                    alpha=0.5)
end
legend = plot([0 0 0], showaxis = false, grid = false, label = ["BBVI" "ADVI" "MCMC"])
Plots.plot(p..., legend, size=(800, 550))
Plots.pdf("./images/rice-lambda")

p = []
for index in L+1:2L
    # Create empty plot
    push!(p, Plots.plot(legendfontsize=4,
                        tickfont=font(4),
                        legend=:none,
                        title=string("\\mu_", index-L),
                        titlefont=font(7),
                        margin=-3mm, fmt=:svg))

    # Add BBVI
    StatsPlots.plot!(LogNormal((Q_bbvi.dists[index].μ),
                            Q_bbvi.dists[index].σ),
                            fill=true,
                            label="BBVI",
                            alpha=0.95)
    # # Add SVGD
    # Plots.histogram!(exp.(X50[:, index]),
    #                 label="SVGD",
    #                 alpha=0.7,
    #                 normalize=true;
    #                 bins=20)
    # Add ADVI
    StatsPlots.plot!(LogNormal((Q_advi.dists[index].μ),
                Q_advi.dists[index].σ),
                fill=true,
                label="ADVI",
                alpha=0.8)

    # Add MCMC with 1000 sample burn-in
    Plots.histogram!(ch.trace[Symbol(string("μ", index-L))][1:end],
                    label="MCMC",
                    normalize=true,
                    bins=30,
                    alpha=0.5)
end
# legend = plot([0 0 0 0], showaxis = false, grid = false, label = ["BBVI" "ADVI" "true value" "MCMC"])
legend = plot([0 0 0], showaxis = false, grid = false, label = ["BBVI" "ADVI" "MCMC"])
Plots.plot(p..., legend)
Plots.pdf("./images/rice-mu")
