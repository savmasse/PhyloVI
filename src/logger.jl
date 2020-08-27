using Plots, StatsPlots
using DataFrames

#===============================================================================
                            ADVI optimization logger
===============================================================================#

# TODO: add read and writing functionality to the Logger
mutable struct VarInfLogger
    df::DataFrame       # DataFrame containing the results for each output
end

function VarInfLogger()
    df = DataFrame()
    return VarInfLogger(df)
end

function VarInfLogger(N, L)

    # Create names
    T = 2*L + N + 1

    # Names for variational mean values
    n = [string("λ", i) for i in 1:L]
    append!(n, [string("μ", i) for i in 1:L])
    append!(n, [string("d", i) for i in 1:N])
    push!(n, string("η"))

    # Names for varaitional standard deviations
    append!(n, [string("σ_λ", i) for i in 1:L])
    append!(n, [string("σ_μ", i) for i in 1:L])
    append!(n, [string("σ_d", i) for i in 1:N])
    push!(n, string("σ_η"))

    # Add the rest of the gradients
    append!(n, [string("g", i) for i in 1:2T])

    # Finally add the ELBO
    push!(n, "ELBO")

    # Create the columns
    df = DataFrame()
    for i in n
        df[Symbol(i)] = Float64[]
    end

    return VarInfLogger(df)
end

function to_param_space(logger::VarInfLogger, q::M, model::P) where {M <: MeanField, P <: PhyloModel}

    # Convert the variational parameters to from real space to param space
    pardf = copy(logger.df)
    qtrans = sample_transform(q)
    mtrans = model_transform(model)
    T = length(q)

    for (i, v) in enumerate(eachrow(pardf))
        x = collect(v)[1:2T]
        x = collect(Iterators.flatten(qtrans(x)))
        μ = x[1:T]
        μ = collect(Iterators.flatten(mtrans(μ)))
        σ = x[T+1:end]
        pardf[i, 1:2T] = [μ..., σ...]
    end

    return pardf
end

function update_logger!(logger::VarInfLogger, θ, ∇, objective)
    push!(logger.df, [θ..., ∇..., objective])
end


################################################################################
#            Various plotting functionality for method comparison
################################################################################

function plot_logs(logger, q, m, num, L, N, T, real_space=true, truth=[])

    # Transform to parameter space if required
    df = (!real_space) ? to_param_space(logger, q, m) : logger.df
    rate_list = sort(StatsBase.sample(collect(1:L), num, replace=false))
    title_font = font(10)
    tick_font = font(6)

    # Create plots for duplication rates
    pλ = []
    for i in rate_list
        toplim = maximum((df[i])) #+ maximum(df[i+T])
        if !isempty(truth); toplim = maximum([toplim, truth[:λ][i]]) end
        p1 = Plots.plot((df[i]),
                        # ribbon=df[i+T],
                        legend=:none,
                        title=string("\\lambda_{",i,"}"),
                        titlefont=title_font,
                        tickfont=tick_font,
                        margin=0mm,
                        top_margin=-3mm)
        if !isempty(truth)
            Plots.hline!([truth[:λ][i]], color=:red)
        end
        push!(pλ, p1)
    end

    # Create plots for loss rates
    pμ = []
    for i in rate_list
        toplim = maximum((df[i+L])) #+ maximum(df[i+L+T])
        if !isempty(truth); toplim = maximum([toplim, truth[:μ][i]]) end
        p1 = Plots.plot((df[i+L]),
                        # ribbon=df[i+L+T],
                        legend=:none,
                        title=string("\\mu_{",i,"}"),
                        titlefont=title_font,
                        tickfont=tick_font,
                        margin=0mm,
                        top_margin=-3mm,
                        ylims=(minimum(df[i+L]), maximum(df[i+L])))
        if !isempty(truth)
            Plots.hline!([truth[:μ][i]], color=:red)
        end
        push!(pμ, p1)
    end

    pd = []
    toplim = maximum((df[T])) + maximum(df[2T])
    # pη = Plots.plot(df[T], ribbon=df[2T], title="\\eta", legend=:none, ylims=(0, toplim))
    pη = Plots.plot(df[T], title="\\eta", legend=:none, ylims=(0, toplim))
    pELBO = Plots.plot(df[end], title="ELBO", legend=:none)

    l = @layout [grid(1,num){0.3h} ; grid(1,num){0.3h} ; [a b]]
    p = [pλ..., pμ..., pd..., pη, pELBO]

    # Plot the full figure
    Plots.plot(p..., layout=l)
end

function compare_methods(Q, model, logger, trace, symbols=[], truth=[], burn_in=1)

    # Setup symbols and names
    symbol_list = names(logger.df)
    p = []

    # Create plot for each symbol
    for s in symbols

        # Sample the MCMC distribution
        mcmc_samples = trace[s][burn_in:end]
        i = findfirst(x->x==s, symbol_list)

        # Convert symbol to latex title
        t = String(s)
        t = replace(t, "λ" => "\\lambda")
        t = replace(t, "μ" => "\\mu")
        t = replace(t, "η" => "\\eta")

        # Apply the model transform to the variational distribution
        d = Normal()
        if typeof(Q) == MeanFieldGaussian
            temp = Distributions.params(Q)[:μ]
            # d = collect(Iterators.flatten(model_transform(model)(temp)))
            d = temp
            d = LogNormal(d[i], Q.dists[i].σ)
        elseif typeof(Q) == FullRankGaussian
            # μ_temp =  collect(Iterators.flatten(model_transform(model)(Q.dists.μ)))
            μ_temp = Q.dists.μ
            σ_temp = diag(Q.dists.Σ.chol.L)
            d = LogNormal(μ_temp[i], σ_temp[i])
        end

        ptemp = StatsPlots.plot(d,
                                label="ADVI",
                                alpha=0.5,
                                normalize=true,
                                title=t,
                                legend=false,
                                tickfont=font(4),
                                fill=true,
                                titlefont=font(8),
                                margin=-1mm,
                                top_margin=-2mm)
        Plots.histogram!(mcmc_samples,
                        label="MCMC",
                        alpha=0.5,
                        normalize=true,
                        legend=false)
        if !isempty(truth) Plots.vline!([truth[i]], color=:red, label="true value") end

        push!(p, ptemp)
    end

    # Add a single invisible subplot to show the legend
    push!(p, Plots.plot((1:3)', legend=:topright, framestyle=:none))

    # num = Int(length(symbols)/2)
    # l = @layout [a; grid(1,num); grid(1,num)]

    # Plots.plot(p[end], p[1:end-1]..., layout=l)
    Plots.plot(p[1:end-1]..., fmt=:svg)
end

# Compare the objective of the MCMC and ADVI
function objective_comparison(logger, chain)

    obj_mcmc = chain.trace[:logp] .+ chain.trace[:logπ]
    obj_advi = logger.df[:ELBO]

    Plots.plot(obj_mcmc,
                label="MCMC (logL + logP)",
                legend=:bottomright,
                title="Objective function comparison",
                titlefont=font(10),
                margin=0mm,
                top_margin=-2mm)
    Plots.plot!(obj_advi, label="ADVI (ELBO)")

end
