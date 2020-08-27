using CSV
using DataFrames

Random.seed!(1)

# species_names = (CSV.read(".\\PhyloVI\\data\\plaza\\species.txt", delim="\t", datarow=1)[1])
# open(".\\PhyloVI\\data\\plaza\\species_list.txt", "w") do f
#     for s in species_names
#         write(f, s, "\n")
#     end
# end

struct GeneFamily
    name::String
    species::Vector{String}
    genes::Vector{String}
end
Base.length(family::GeneFamily) = length(family.genes)
function GeneFamily(name::String, species::String, gene::String)
    family = GeneFamily(name, Vector{String}([species]), Vector{String}([gene]))
    return family
end
function addgene(family::GeneFamily, species::String, gene::String)
    push!(family.species, species)
    push!(family.genes, gene)
end
function filter_size(gf::Dict{String, GeneFamily}, min_size::Int, max_size::Int)
    v = Vector{GeneFamily}()
    for family in values(gf)
        if length(family) >= min_size && length(family) <= max_size
            push!(v, family)
        end
    end
    return v
end
function getcounts(family::GeneFamily, species_list::AbstractVector{String})
    # Init dictionary
    counts = Dict(s => 0 for s in species_list)

    # Go through genes in the family and update species counts
    for index in eachindex(family.genes)
        s = family.species[index]
        counts[s] += 1
    end
    return [counts[s] for s in species_list]
end

function getdataframe(families::AbstractVector{GeneFamily}, species_list::AbstractVector{String})
    # Create dataframe
    df = DataFrame(repeat([Int[]], length(species_list)))
    # Now fill the dataframe with the gene count data
    for family in families
        push!(df, getcounts(family, species_list))
    end
    rename!(df, species_list)
    return df
end

# Remove a certain species from the dataframe
function rm_species!(df::DataFrame, species_name::String)
    sym_name = Symbol(species_name)
    delete!(df, sym_name)
end

function loaddata(df::DataFrame)
    # Load all data into dictonaries of GeneFamilies
    gf = Dict{String, GeneFamily}()
    for index in 1:size(df)[1]
        f, s, g = df[index, :]
        if haskey(gf, f)
            addgene(gf[f], s, g)
        else
            gf[f] = GeneFamily(f, s, g)
        end
    end
    return gf
end


# Load the data from the PLAZA gene family file
# tree = open(".\\PhyloVI\\data\\plaza\\tree_dicots_4.0.nwk", "r") do f ; readline(f); end
# df = CSV.read(".\\PhyloVI\\data\\plaza\\species_list_dicots_4.0_partial.txt", delim=',', datarow=1)

# tree = open(".\\PhyloVI\\data\\plaza\\species_list3.nwk", "r") do f ; readline(f); end
tree = open(".\\PhyloVI\\data\\plaza\\tree.nwk", "r") do f ; readline(f); end
# df = CSV.read(".\\PhyloVI\\data\\plaza\\species_list2_partial.txt", delim=',', datarow=1)
df = CSV.read(".\\PhyloVI\\data\\plaza\\species_list_dicots_4.0_partial.txt", delim=',', datarow=1)

species_dict = Dict(df[i,:][2] => df[i,:][1] for i in 1:size(df)[1])
df = CSV.read(".\\PhyloVI\\data\\plaza\\genefamily_data.orth.csv", delim="\t")
og = unique(df[:gf_id])
species_list = unique(df[:species])
# tree_species = [i for i in keys(species_dict)]

# Process raw data into GeneFamilies
gf = loaddata(df)

# Do some size filtering
families = filter_size(gf, 30, 200)
df = getdataframe(families, species_list)

# Change the names of the columns to the names used in the tree...
rename!(df, species_dict)

# Filter some of the PLAZA species out that are not in the tree database
filter_list =   ["Arachis_ipaensis",
                "Chenopodium_quinoa",
                "Corchorus_olitorius",
                "Micromonas_commoda",
                "Utricularia_gibba",
                "Ziziphus_jujuba"]
for species in filter_list
    try
        rm_species!(df, species)
    catch e
        nothing
    end
end

# Make sure that is at least one count in the outgroup
spec = collect(values(species_dict))
# Atr must be filtered when using species_list2
# df = filter(row -> row.Amborella_trichopoda > 0, df)
# df = filter(row -> row.Picea_abies > 0, df)
# df = filter(row -> row.Physcomitrella_patens > 0, df)
#
# # Remove Chlamydomonas reinhardtii
# df = filter(row -> row.Chlamydomonas_reinhardtii > 0, df)

function filter_all(df)
    for r in names(df)
        df = filter(row -> row[r] > 0, df)
    end
    return df
end

df = filter_all(df)


# s = open(".\\PhyloVI\\data\\plaza\\species_list3.txt") do file
#     readlines(file)
# end
# s = [Symbol(replace(i, (" "=>"_"))) for i in s]
# df = df[s]

# Take a subsample of the full dataset
indices = rand(1:size(df)[1], 250)
df = df[indices, :]

# Run a test on this new tree...
# Load the tree
# datadir = "./PhyloVI/data/"
# tree = open(joinpath(datadir, "plaza/tree.nwk"), "r") do f ; readlines(f); end
# tree = join(tree)

# Init model
λ, μ, η = 2., 2., .9
model, profile = DLWGD(tree, df, λ, μ, η)

# Create a prior
data = profile
prior = IidRevJumpPrior(
    Σ₀=[0.5 0.45 ; 0.45 0.5],
    X₀=MvNormal(log.(ones(2)), [0.5 0.45 ; 0.45 0.5]),
    πK=DiscreteUniform(0,20),
    πq=Beta(1,1),
    πη=Beta(3,1),
    Tl=treelength(model))

# Create a Beluga model with a WGD
bm = WGDBelugaModel(prior, model)
N = length(Beluga.getwgds(model))
L = length(model) - 2N
T = 2*L + N + 1         # (L x λ, L x μ, N x q, and 1 x η)
x = [Normal(log(.1), 0.05) for _ in 1:T-1]
push!(x, Normal(0.1, 0.05))
q = MeanFieldGaussian(x)

elbo = ELBO(1)
# advi = ADVI(1, 5, 2, 10, 1e-2, VarInfLogger(N, L))
# opt = ADAM(0.02)
# Q = optimize(advi, elbo, q, bm, data, opt)
#
# r = Distributions.params(Q)[:μ]
# res = model_transform(bm)(r)


# Do same with BBVI
x = [Normal(-5, 0.01) for _ in 1:T-1]
push!(x, Normal(.5, 0.01))
q = MeanFieldGaussian(x)

Random.seed!(1)
opt = ADAM(0.01)
logger_bbvi = VarInfLogger(N, L)
@time Q_bbvi = optimize(1000, 10, q, bm, opt, data, 30, logger_bbvi)
res_bbvi = Distributions.params(Q_bbvi).μ
μ_bbvi = model_transform(bm)(res_bbvi)
σ_bbvi = Distributions.params(Q_bbvi).σ
ζ_bbvi = collect(logger_bbvi.df[end, :])

Plots.plot(logger_bbvi.df[:η])



L = 89
p = []
for index in L+1:2L
    # Create empty plot
    push!(p, Plots.plot(legendfontsize=4,
                        tickfont=font(4),
                        legend=:none,
                        title=string("\\mu_", index-L),
                        titlefont=font(7),
                        margin=-5mm, fmt=:svg))

    # Add BBVI
    StatsPlots.plot!(LogNormal((Q_bbvi.dists[index].μ),
                            Q_bbvi.dists[index].σ),
                            fill=true,
                            label="BBVI",
                            alpha=0.95)

    # Add ADVI
    # StatsPlots.plot!(LogNormal((Q_advi.dists[index].μ),
    #             Q_advi.dists[index].σ),
    #             fill=true,
    #             label="ADVI",
    #             alpha=0.8)
end
# legend = plot([0 0 0 0], showaxis = false, grid = false, label = ["BBVI" "ADVI" "true value" "MCMC"])
legend = plot([0 0 0], showaxis = false, grid = false, label = ["BBVI" "ADVI" "MCMC"])
Plots.plot(p..., legend)


p = [collect(Iterators.flatten(model_transform(bm, i))) for i in rand(Q_bbvi, 5_000)]
df = DataFrame(transpose(hcat(p...)))
StatsPlots.violin([[i] for i in 1:L], [df[i] for i in 1:L],
                                marker=(0.2,:blue,stroke(0)),
                                xticks=1:L, legend=:none,
                                color="red", alpha=0.8,
                                tickfont=font(4);
                                ylabel="\\lambda")
Plots.pdf("./images/plaza-lambda")

p = [collect(Iterators.flatten(model_transform(bm, i))) for i in rand(Q_bbvi, 5_000)]
df = DataFrame(transpose(hcat(p...)))
StatsPlots.violin([[i] for i in 1:L], [df[i] for i in L+1:2L],
                                marker=(0.2,:blue,stroke(0)),
                                xticks=1:L, legend=:none,
                                color="red", alpha=0.8,
                                tickfont=font(4);
                                ylabel="\\mu")
Plots.pdf("./images/plaza-mu")
