
using Beluga, CSV, DataFrames, Parameters
using Measures, IterTools, BenchmarkTools

using PhyloVI

include("../src/vimodel.jl")
include("../src/phylomodel.jl")
include("../src/meanfield.jl")
include("../src/elbo.jl")
include("../src/logger.jl")
include("../src/advi.jl")
include("../src/wgdbelugamodel.jl")

# Set seed for reproducibility
Random.seed!(1)

# get some data
datadir = "./PhyloVI/data/"
tree = open(joinpath(datadir, "species_trees/plants2.nw"), "r") do f ; readline(f); end
df = CSV.read(joinpath(datadir, "branch_wise/1.counts.csv"), delim=",")

# Init model
λ, μ, η = 1., 1., 0.5
model, profile = DLWGD(tree, df, λ, μ, η)

# insertwgd!(model, model[6], 0.05, 0.25)
# extend!(profile, 6);

# Simulate some more data
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

# Create a Beluga model with a WGD
bm = WGDBelugaModel(prior, model)
N = length(Beluga.getwgds(model))
L = length(model) - 2N
T = 2*L + N + 1         # (L x λ, L x μ, N x q, and 1 x η)
xn = [Normal(log(1), 0.05) for _ in 1:T-1]
push!(xn, Normal(1., .05))
q = MeanFieldGaussian(xn)

# Run ADVI algorithm
elbo = ELBO(1)
advi = ADVI(1, 500, 2, 200, 1e-4, VarInfLogger(N, L))
@time Q = optimize(advi, elbo, q, bm, data, 0.1, 0.05)
# Save VI values to csv
CSV.write("./results/simulation_results_vi_α=0.9_η=0.1.csv", advi.logger.df)

# Compare with simulated values
r = Distributions.params(Q)[:μ]
res = model_transform(bm)(r)
rr = CSV.read(".\\PhyloVI\\data\\branch_wise\\1.rates.csv", delim=",")
d = DataFrame((λ_truth=rr[:λ], λ=res[:λ], μ_truth=rr[:μ], μ=res[:μ]))

# Now run the MCMC equivalent on the same data
ch = RevJumpChain(data=data, model=deepcopy(model), prior=prior)
init!(ch)
rjmcmc!(ch, 100, show=10, trace=1)

# Save MCMC values to csv
CSV.write("./results/simulation_results_mcmc_α=0.9_η=0.1.csv", ch.trace)

# Plot the logs and compare to the true values
plot_logs(advi.logger, q, bm, 5, L, N, T, false)
# savefig("beluga_simulation_advi_2.pdf")

all_rates = [Symbol(string("λ", i)) for i in 1:L]
# append!(all_rates, [Symbol(string("μ", i)) for i in 1:L])
compare_methods(Q, bm, advi.logger, ch.trace, all_rates, x, 1)
