using Distributions, Test, DynamicHMC
using Whale
using Whale.NewickTree: postwalk, getlca, insertnode!

t = deepcopy(Whale.extree)
n = length(postwalk(t))
insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_1")

r = RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2], η=0.9))
w = WhaleModel(r, t, 0.1)
D = read_ale(joinpath(@__DIR__, "../..//dev/Whale.jl/example/example-1/ale"), w, true)
prior = CRPrior(MvLogNormal(ones(2)), Beta(3,1), Beta())
problem = WhaleProblem(D, w, prior)
p, ∇p = Whale.logdensity_and_gradient(problem, zeros(4))
shouldbe = [-276.061487822, 160.141680772, -7.199728049, -1.26771978]
@test p ≈ -462.771909465
@test all(∇p .≈ shouldbe)
