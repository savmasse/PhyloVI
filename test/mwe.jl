
using Beluga, CSV, DataFrames, Parameters
using ForwardDiff

# Load tree and dataframe
datadir = "test/data"
tree = open(joinpath(datadir, "plants1.nw"), "r") do f ; readline(f); end
df = CSV.read(joinpath(datadir, "plants1-100.tsv"), delim=",")

# Get model and data
λ, μ, η = 1.0, 0.92, 0.66
model, profile = DLWGD(tree, df, λ, μ, η)

prior = IidRevJumpPrior()
logpdf(prior, model)    # Logprior OK!

# Function to update all parameters
function update_all!(model, x)
    L = length(model)
    μ = x[1:L]
    λ = x[L+1:2L]
    η = x[end]
    update!(model[1], (λ=λ[1], μ=μ[1], η=η))    # ForwardDiff fails here...
    for i in 2:L
        update!(model[i], (λ=λ[i], μ=μ[i]))
    end
end

function grad_logprior(prior, model, new_params)

    function helper_function(x)
        update_all!(model, x)
        return logpdf(prior, model)
    end

    return ForwardDiff.gradient(helper_function, new_params)
end

# Get some new parameters for gradient calculation
θ = rand(length(asvector(model)))
grad_logprior(prior, model, θ)      # This breaks on ForwardDiff...
