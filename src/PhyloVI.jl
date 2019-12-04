module PhyloVI

    using Distributions
    using LinearAlgebra
    using TransformVariables
    using Parameters
    using DataFrames
    using ForwardDiff
    using CSV

    include("meanfield.jl")
    include("elbo.jl")
    include("advi.jl")

    export
    MeanFieldGaussian,
    ELBO,
    elbo,
    sample_transform
end
