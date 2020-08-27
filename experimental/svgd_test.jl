include("svgd.jl")


# Set seed for reproducibility
Random.seed!(1)

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

# Create a Beluga model with a WGD
bm = WGDBelugaModel(prior, model)
N = length(Beluga.getwgds(model))
L = length(model) - 2N
T = 2*L + N + 1
t = model_transform(bm)

# Create a suitable initial distribution
opt = ADAM(0.1)
s = SVGD(1, 100, opt, -1, VarInfLogger())
# m = MvLogNormal(repeat([log(log(3))], T), ones(T)/5)
m = MvNormal(-ones(T), ones(T))
x0 = rand(m, s.n_particles)
x0 = convert(Array{Float64, 2}, transpose(x0))

# Do optimization
x = optimize(s, bm, data, 100, x0)
xt = [collect(Iterators.flatten(t(xi))) for xi in eachrow(x)]
xt = transpose(hcat(xt...))

p = []
symbols = [Symbol(string("λ", i)) for i in 1:L]
burnin = -1
truth = [rr[:λ]..., rr[:μ]..., η]
for i in 1:L#size(x)[2]
    y = KernelDensity.kde(xt[:, i])
    pt = Plots.plot(y, legend=:none)
    Plots.histogram!(xt[:, i],
                     alpha=0.4,
                     normalize=true,
                     tickfont=font(5),
                     yticks=:none,
                     margin=0mm)
    Plots.vline!([truth[i]])

    # If MCMC available, add to the plot for comparison
    if burnin > 0
        Plots.histogram!(ch.trace[symbols[i]][burnin:end],
                         normalize=true,
                         alpha=0.6)
    end
    push!(p, pt)
end

function svgd_score(x, model, data)
    res = zero(Float64)
    t = model_transform(model)
    for ζ in eachrow(x)
        θ = t(ζ)
        m = model(θ)
        res += logprior(m) + loglikelihood(m, data) + TransformVariables.transform_and_logjac(t, ζ)[2]
    end
    return res/size(x)[1]
end

Plots.plot(p...)

# svgd_score(x0, bm, data)
# svgd_score(x, bm, data)

###############################################################################
#####      Test new SVGD functionality         ################################
###############################################################################
#
# K = 2
# μ = [-5., 5.]
# σ = [0.5, 0.2]
# π = [0.3, 0.7]
# θ = [μ, σ, π]
# true_model = GaussianMixtureModel(K, μ, σ, π)
# data = rand(true_model, 500)
#
# # Test the Guassian Mixture model
# model = GaussianMixtureModel(K, zeros(K), ones(K), repeat([1/K], K))
# s = SVGD(30, 500, ADAM(.2), -1., VarInfLogger())
# t = model_transform(model)
# d = dimension(t)
# m = MvNormal(ones(d), [2., 2., 0.1, 0.1, 0.1])
# x0 = rand(m, s.n_particles)
# x0 = convert(Array{Float64, 2}, transpose(x0))
#
# # Get result in parameter space
# x = optimize(s, model, data, 10, x0)
# xt = [collect(Iterators.flatten(t(xi))) for xi in eachrow(x)]
# xt = transpose(hcat(xt...))
#
# names = [:m1, :m2, :o1, :o2, :p1, :p2]
# p = []
# for i in 1:size(xt)[2]
#     y = kde(xt[:, i])
#     pt = Plots.plot(y, yticks=:none, legend=:none, title=names[i])
#     Plots.histogram!(xt[:, i], alpha=0.4, normalize=true, bins=20)
#     push!(p, pt)
# end
#
# Plots.plot(p...)
