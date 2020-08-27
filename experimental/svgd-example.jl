include("svgd.jl")
using Flux.Optimise

gmm = GaussianMixtureModel(1, [1.], [.5], [1.])
data = rand(gmm, 1000)
N=500

Random.seed!(1)
opt = ADAM(0.1)
svgd = SVGD(N, 0, opt, -1, VarInfLogger())
X1 = optimize(svgd, gmm, data)

Random.seed!(1)
opt = ADAM(0.1)
svgd = SVGD(N, 10, opt, -1, VarInfLogger())
X10 = optimize(svgd, gmm, data)

Random.seed!(1)
opt = ADAM(0.1)
svgd = SVGD(N, 50, opt, -1, VarInfLogger())
X100 = optimize(svgd, gmm, data)

ph = Plots.histogram(X1[:, 1], label="N=0",
                normalize=true, alpha=0.8,
                fmt=:svg, title="Final SVGD distribution by number of iterations N";
                titlefontsize=11, xlabel="\\mu")
Plots.histogram!(X10[:, 1], label="N=10", normalize=true, alpha=0.8)
Plots.histogram!(X100[:, 1], label="N=50", normalize=true, alpha=0.8)

Plots.pdf(ph, "svgd-example")


μ = []
σ = []
for i in 1:100
    Random.seed!(1)
    opt = ADAM(0.1)
    svgd = SVGD(1, i, opt, -1, VarInfLogger())
    X = optimize(svgd, gmm, data)
    push!(μ, X[:, 1][1])
    push!(σ, exp(X[:, 2][1]))
end

p = Plots.plot(μ, label="\\mu", title="Single-particle SVGD", fmt=:svg, ylabel="parameter value", xlabel="# iterations")
Plots.hline!([1.], linestyle=:dot, color="blue", label="")
Plots.plot!(σ, label="\\sigma")
Plots.hline!([0.5], linestyle=:dot, color="green", label="")
Plots.pdf(p, "svgd-map")

Plots.plot(ph)
