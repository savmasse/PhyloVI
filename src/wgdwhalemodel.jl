using TransformVariables
using Whale
using Whale.NewickTree: postwalk, getlca, insertnode!
using Distributions

struct WGDWhaleModel <: PhyloModel
        prior
        model # TODO: change this name because we usually end up writing model.model which is silly...
        tree
        Δ
end
Base.length(model::WGDWhaleModel) = length(model.model)

function WGDWhaleModel(prior, model, tree)
        return WGDWhaleModel(prior, model, tree, 0.1)
end

function (model::WGDWhaleModel)(nt::NamedTuple)

        # Get the available keys and fill in dummy values for unavailable
        # p = (   λ = nt.λ,
        #         μ = nt.μ,
        #         q = haskey(nt, :q) ? nt.q : Float64[],
        #         p = haskey(nt, :p) ? nt.p : Float64[],
        #         η = nt.η        )

        t = typeof(model.model.params)
        pp = properties(model.model.params).names
        v = []
        for prop in pp
                if hasproperty(nt, prop)
                        push!(v, getfield(nt, prop))
                else
                        push!(v, Float64[]) # Add empty array
                end
        end
        p = NamedTuple{Tuple(collect(pp))}(v)

        # Create a new RatesModel of correct type
        r = RatesModel(t(p...))
        return WGDWhaleModel(model.prior, r, model.tree, model.Δ)
end

function (model::WGDWhaleModel)(x::AbstractVector{T}) where T<:Real
        m = model.model.params
        pp = properties(m).names
        s = []
        v = []
        i = 0
        for prop in pp
                propval = getproperty(m, prop)
                push!(s, prop) # Add name
                push!(v, (prop != :η) ? x[i+1:length(propval)] : x[i+1]) # Add values
                i += length(propval)
        end
        p = NamedTuple{Tuple(s)}(v)

        # Create a new RatesModel of correct type
        t = typeof(model.model.params)
        r = RatesModel(t(p...))

        return WGDWhaleModel(model.prior, r, model.tree, model.Δ)
end

function params(model::WGDWhaleModel)
        # Filter out the empty properties
        k = []
        v = []
        for prop in keys(model.model.trans.transformations)
                propval = getproperty(model.model.params, prop)
                if length(propval) != 0
                        push!(k, prop)
                        push!(v, propval)
                end
        end
        return NamedTuple{Tuple(k)}(v)
end

function logprior(model::WGDWhaleModel)
        return logpdf(model.prior, model.model)
end

function grad_logprior(model::WGDWhaleModel)
        θ = params(model)
        ζ = model_invtransform(model)(θ)
        _, ∇f =  Whale.fand∇f(model.prior, model.model, ζ)
        return ∇f
end

function loglikelihood(model::WGDWhaleModel, data::AbstractArray)
        w = WhaleModel(model.model, model.tree, model.Δ)
        return Whale.logpdf(w, data)
end

function grad_loglikelihood(model::WGDWhaleModel, data::AbstractArray)
        θ = params(model)
        ζ = model_invtransform(model)(θ)
        w = WhaleModel(model.model, model.tree, model.Δ)
        _, ∇f = Whale.fand∇f(w, data, ζ)
        return ∇f
end

function model_transform(model::WGDWhaleModel)

        # Need to check if there are empty fields they error on the inverse
        # This is presumable a bug with TransformVariables

        k = []
        t = []
        trans = model.model.trans
        transformed = trans(rand(dimension(trans)))
        for key in keys(trans.transformations)
                if length(getproperty(transformed, key)) != 0
                        push!(k, key)
                        push!(t, getproperty(trans.transformations, key))
                end
        end

        nt = NamedTuple{Tuple(k)}(t)
        return TransformVariables.TransformTuple(nt)
end

function model_invtransform(model::WGDWhaleModel)
        return inverse(model_transform(model))
end




# Try some stuff
t = deepcopy(Whale.extree)
n = length(postwalk(t))
insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_1")

r = RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2], η=0.9))
w = WhaleModel(r, t, 0.1)
data = read_ale(joinpath(@__DIR__, "../../dev/Whale.jl/example/example-1/ale"), w, true)
prior = CRPrior(MvLogNormal(ones(2)), Beta(3,1), Beta())
problem = WhaleProblem(data, w, prior)
p, ∇p = Whale.logdensity_and_gradient(problem, zeros(4))

model = WGDWhaleModel(prior, r, t)
logprior(model)
loglikelihood(model, data)


a = rand(4)
b = collect(Iterators.flatten(model_transform(model, a)))
m = model(b)

logprior(m)
loglikelihood(m, data)
grad_logprior(m)
grad_loglikelihood(m, data)
Whale.fand∇f(prior, m.model, a)
wm = WhaleModel(m.model, m.tree, 0.1)
Whale.fand∇f(wm, data, a)

model(params(model))
Whale.logpdf(wm, data)
logprior(m)

r = RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2], η=0.9))
w = WhaleModel(r, t, 0.1)
prior = CRPrior(MvLogNormal(ones(2)), Beta(3,1), Beta())
problem = WhaleProblem(data, w, prior)

y = rand(4)
x = w.rates.trans(y)
Whale.logpdf(w(x), data) == Whale.fand∇f(w(x), data, y)[1]
Whale.logpdf(w(x), data)
Whale.fand∇f(w(x), data, y)[1]


# Test ADVI
opt = ADAM(.01)
x = [Normal(log(1.), 0.05) for _ in 1:dimension(model_transform(model))]
q = MeanFieldGaussian(x)
elbo = ELBO(1)
logger = VarInfLogger(DataFrame(repeat([Float64[]], 17)))
advi = ADVI(1, 300, 2, 12, 10^-3, logger)
Q = optimize(advi, elbo, q, model, data, opt)

StatsPlots.plot(LogNormal(exp(Q.dists[1].μ), Q.dists[1].σ))
