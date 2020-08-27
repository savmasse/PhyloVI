
function grad_logprior(model::WGDBelugaModel)

    function helper(ζ)
        θ = collect(Iterators.flatten(model_transform(model, ζ)))
        return logprior(model(θ))
    end

    θ = params(model)
    ζ = model_invtransform(model)(θ)
    return ForwardDiff.gradient(helper, ζ)
end

function grad_loglikelihood2(model::WGDBelugaModel, data::AbstractArray{T}) where T

    function helper(ζ, x)
        θ = collect(Iterators.flatten(model_transform(model, ζ)))
        return logpdf(model(θ).model, x.xp)
    end
    g(x) = ForwardDiff.gradient(ζ->helper(ζ, x), ζ)

    θ = params(model)
    ζ = model_invtransform(model)(θ)
    return mapreduce(g, +, data)
end

d = data[1]
logprior(bm)
grad_logprior(bm)

ζ = rand(T)
θ = model_transform(bm, ζ)
m = bm(θ)
g1 = grad_loglikelihood(m, data) .* grad_invtransform(m, ζ)
g2 = grad_loglikelihood2(m, data)
