module PhyloVI

    using Distributions
    using LinearAlgebra
    using TransformVariables
    using Parameters

    include("meanfield.jl")
    include("elbo.jl")
    include("advi.jl")

    export
    MeanFieldGaussian,
    ELBO,
    elbo,
    sample_transform
end
