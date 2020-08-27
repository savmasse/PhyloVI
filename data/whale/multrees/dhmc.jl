using Pkg; Pkg.activate("/home/arzwa/dev/Whale/docs")
using Whale
using Whale.NewickTree, Distributions, DynamicHMC, DataFrames, LinearAlgebra, Random, CSV
using DynamicHMC.Diagnostics

trees = readnw.(readlines(joinpath(@__DIR__, "data/trees.nw")))
for tree in trees, n in prewalk(tree)
    n.data.distance = isroot(n) ? NaN : 1.
end
# trees 2,3 and 4 are MUL trees

# This uses a subset of 30 families from the data in
# `data/syn-clusters.ale.tar.gz`
aledir = joinpath(@__DIR__, "data/everywhere.ale/")
problems = map(trees) do tree
    n = length(prewalk(tree))
    rates = RatesModel(Whale.Critical(
        λ=randn(n), η=rand()), fixed=(:p,))
    model = WhaleModel(rates, tree, 0.1)
    data  = read_ale(aledir, model, true)
    prior = Whale.ExpPrior(
        πη=Beta(9,1),
        πr=Exponential(0.1))
    WhaleProblem(data, model, prior)
end
# I had issues with η -> 0. for trees 3 & 4, so I put a fairly strong
# prior that corresponds to more or less the posterior for η for tree 1

harmmean(xs) = mean(xs .^-1)^-1

function get_priorp(problem, posterior)
    map(posterior) do θ
        r = problem.model(θ).rates
        logpdf(problem.prior, r)
    end
end

results = map(problems) do problem
    # sample with NUTS
    results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100,
        warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=2))
    @info summarize_tree_statistics(results.tree_statistics)
    posterior = Whale.transform(problem, results.chain)
    # obtain the unnormalized posterior values and prior logpdf
    πs = map(x->x.π, results.tree_statistics)
    ps = get_priorp(problem, posterior)
    ls = πs .- ps  # recompute the likelihood
    (df=Whale.unpack(posterior), hm=harmmean(ls))
end

@info map(x->x.hm, results)



# Try VI modelling on small subset of data
res = []
models = []
datas = []
indices = [1, 4]
for index in indices

    # Get problem details
    problem = problems[index]
    data = problem.data
    tree = trees[index]
    wm = problem.model.rates
    prior = problem.prior
    model = WGDWhaleModel(prior, wm, tree)

    # Setup ADVI
    N = dimension(wm.trans)
    L = 4N + 1
    q = MeanFieldGaussian([-ones(N)..., ones(N)/100...])
    elbo = ELBO(1)
    logger = VarInfLogger(DataFrame(repeat([Float64[]], L)))
    # opt = ADAM(0.01)
    # advi = ADVI(1, 10, 1, length(data), 0.01, logger) # No subsampling for now

    # Optimize
    # Q = optimize(advi, elbo, q, model, data, opt)

    opt = ADAM(0.05)
    logger_bbvi = VarInfLogger(DataFrame(repeat([Float64[]], L)))
    Q = optimize(500, 20, q, model, opt, data, 100, logger)

    # Show result
    push!(res, Q)
    push!(models, model)
    push!(datas, data)
end

# Calculate high-quality final ELBO
for index in 1:length(res)
    elbo = ELBO(100)
    println("Final ELBO for tree ", index, " : ", 13 / (dimension(models[index].model.trans)-1) * elbo(res[index], models[index], datas[index]))
end


# SHOULD THE ELBO function be normalized for the amount of branches ???
