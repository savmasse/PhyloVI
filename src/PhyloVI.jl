module PhyloVI

    using Distributions
    using LinearAlgebra
    using TransformVariables
    using Parameters
    using DataFrames
    using ForwardDiff
    using CSV
    using Distributions
    using Beluga

    include("vimodel.jl")
    include("phylomodel.jl")
    include("belugamodel.jl")
    include("wgdbelugamodel.jl")
    include("meanfield.jl")
    include("elbo.jl")
    include("logger.jl")
    include("advi.jl")

    export
    MeanFieldGaussian, ELBO, elbo, sample_transform
end
