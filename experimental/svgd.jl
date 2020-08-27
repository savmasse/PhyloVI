using KernelDensity
using StatsPlots
using Plots
using Flux.Optimise

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


struct SVGD <: VariationalAlgorithm
    n_particles::Int
    n_iterations::Int
    optimizer # Flux optimizers don't have a type...
    σ::Float64
    logger::VarInfLogger
end
SVGD() = SVGD(10, 100, ADAM(0.1), VarInfLogger())

function optimize(svgd::SVGD, model::T, data::AbstractArray, batch_size=-1, x0=nothing) where T<:VIModel

    if batch_size < 0; batch_size = size(data)[1]; end
    t = model_transform(model)
    if isnothing(x0)
        x0 = rand(svgd.n_particles, dimension(t))
    end
    x = copy(x0)

    # Set up the model gradient calculation
    function ∇p(x::AbstractVector{T}, batch::AbstractArray, batch_factor::Float64) where T<:Real
        θ = collect(Iterators.flatten(t(x)))
        # return grad_logprior(model(θ)) + grad_loglikelihood(model(θ), batch) * batch_factor
        m = model(θ)
        ζ = copy(x)
        return (grad_logprior(m) .+ grad_loglikelihood(m, batch) .* batch_factor) .+ grad_logdetjac(m, ζ)
    end

    for i in 1:svgd.n_iterations

        # Sample minibatch and convert back to distributed object
        batch = StatsBase.sample(data, batch_size, replace=false)
        batch = distribute(collect(batch))
        batch_factor = size(data)[1] / batch_size

        # Get the kernel and kernel gradient
        K, ∇K = svgd_kernel(svgd, x)

        # Calculate the model gradients for each particle
        ∇P = [∇p(xj, batch, batch_factor) for xj in eachrow(x)]
        ∇P = transpose(hcat(∇P...))

        # Calculate particle gradient
        ∇x = K * ∇P + ∇K
        ∇x /= svgd.n_particles

        # Perform optimization
        Optimise.update!(svgd.optimizer, x, -∇x)
        score = median(abs.(∇x))

        # score = svgd_score(x, model, data)
        update_logger!(svgd.logger, [x[1, :]..., x[1, :]...], [∇x[1, :]..., ∇x[1, :]...], score)
        println("iteration ", i, ": ", score)
    end

    # Return the final distribution
    return x
end

function svgd_kernel(svgd::SVGD, x::AbstractArray{T}) where T<:Real

    # Pairwise euclidean distance between particles
    E = [(norm(i-j, 2)^2) for i in eachrow(x), j in eachrow(x)]

    # If σ < 0 then use median trick
    σ = svgd.σ
    if σ < 1
        σ = median(E)
        σ = sqrt((σ/2) / log(size(x)[1] + 1))
        if σ == 0; σ = 1.0; end
    end

    # Calculate the RBF kernel
    K = exp.(-1/(2*σ^2) .* E)

    # Calculate kernel gradient
    ∇K = -K * x
    s = sum(K, dims=2)
    for i in 1:size(x)[2]
        ∇K[:, i] += x[:, i] .* s
    end
    ∇K /= σ^2

    return K, ∇K
end
