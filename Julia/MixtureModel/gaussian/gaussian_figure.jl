using Distributions: Gamma, Categorical
using Gen: logsumexp
using CSV, DataFrames
import StatsBase
using Serialization

include("../shared.jl")
include("../dpmm-data-structures.jl")
include("gaussian-clusters-fixedstd.jl")
include("../locally-optimal-smc.jl")
include("../smcp3.jl")
include("gaussian-visualization.jl")
include("utils.jl")
# normal(args...) = rand(Normal(args...))
mean(v) = sum(v)/length(v)
# colors() = ColorSchemes.tol_bright

generate_synthetic_dataset(hypers, alpha, n) = generate_synthetic_trace(hypers, alpha, n).data
generate_synthetic_trace(hypers, alpha, n; sorted=false) = generate_synthetic_fsgaussian_trace(hypers, alpha, n; sorted)
REPS = 10
N = 10

HYPERS = FSGaussianHyperparameters(0, 1_000_000, 1)
ALPHA = 3.0

# Uncomment this to generate a dataset:
# HYPERS = FSGaussianHyperparameters(0, 6, 1)
# gt_trace = generate_synthetic_trace(HYPERS, ALPHA, 100; sorted=false)
# dataset = gt_trace.data 

# Or load this dataset, which is visualized in the paper:
dataset = deserialize("Julia/MixtureModel/datasets/dataset_from_fig8.jld");

HYPERS = FSGaussianHyperparameters(0, 1_000_000, 1) # Inference hyperparameters
gibbs_returns = [run_gibbs_smc(dataset, N, HYPERS, FSGaussianCluster, ALPHA, LocallyOptimalSMCOptions(10, 0, N/5); return_record=true) for _=1:REPS];
smcp310_returns = [run_smcp3(dataset, N, HYPERS, FSGaussianCluster, ALPHA, SMCP3Options(N/5, 10); return_record=true) for _ in 1:REPS];
gibbs_traces = resample.(map(first , gibbs_returns))
smcp310_traces = resample.(map(first , smcp310_returns));

gibbs_smc_results = [logmeanexp(res[1][2]) for res in gibbs_returns]
smcp310_results = [logmeanexp(res[1][2]) for res in smcp310_returns]
println("Gibbs SMC results (synthetic, challenging, $N particles): $(StatsBase.mean(gibbs_smc_results)) +/- $(StatsBase.std(gibbs_smc_results))")
println("SMCP3 results (synthetic, challenging, $N particles, 10 split particles): $(StatsBase.mean(smcp310_results)) +/- $(StatsBase.std(smcp310_results))")

function get_logz_estimates_over_time(results)
    logweight_list = map(last ∘ last, results) # logweight_list[i][t][j] = jth particle logweight at time t from replicate i
    # logweight_list = map(first, weighted_particle_lists)
    time_to_avg_logweight = []
    for t=1:length(logweight_list[1])    
        log_avg_particle_weights = [
            logmeanexp(logweight_list[i][t])
            for i=1:length(logweight_list)
        ]
        avg_log_est = mean(log_avg_particle_weights)
        push!(time_to_avg_logweight, avg_log_est)
    end
    return time_to_avg_logweight
end

f = Figure();
ax = Axis(f[1, 1])
logz_gibbs = get_logz_estimates_over_time(gibbs_returns);
logz_smcp3 = get_logz_estimates_over_time(smcp310_returns);
# CairoMakie.scatter!(ax, 1:length(logz_gibbs), convert(Vector{Float32}, [x/i for (i, x) in enumerate(logz_smcp3)]))
# CairoMakie.scatter!(ax, 1:length(logz_gibbs), convert(Vector{Float32}, [x/i for (i, x) in enumerate(logz_gibbs)]))
CairoMakie.scatter!(ax, 1:length(logz_gibbs), convert(Vector{Float32}, logz_smcp3))
CairoMakie.scatter!(ax, 1:length(logz_gibbs), convert(Vector{Float32}, logz_gibbs))
f

function plot_mean_n_clusters(time_to_particles)
    f = Figure()
    ax = Axis(f[1, 1])
    meann(t) = mean([length(trace.clusters) for trace in time_to_particles[t]])
    CairoMakie.lines!(ax, 1:length(time_to_particles), [meann(t) for t=1:length(time_to_particles)])
    return f
end
function plot_many_mean_n_clusters(coloridx_to_run_to_time_to_particles, colors)
    f = Figure()
    ax = Axis(f[1, 1])
    plot_many_mean_n_clusters!(ax, coloridx_to_run_to_time_to_particles, colors)
    return f
end
function plot_many_mean_n_clusters!(ax, coloridx_to_run_to_time_to_particles_and_logweights, colors)
    for (coloridx, color) in enumerate(colors)
        for run in coloridx_to_run_to_time_to_particles_and_logweights[coloridx]
            time_to_particles, time_to_logweights = run
            style_ax!(ax, length(time_to_particles))
            particle_probs(t) = exp.(time_to_logweights[t] .- logsumexp(time_to_logweights[t]))
            meann(t) = sum([prob * length(trace.clusters) for (prob, trace) in zip(particle_probs(t), time_to_particles[t])])
            # println("meann(1) = $(meann(1))")
            # if coloridx == length(colors)
                # CairoMakie.lines!(ax, 1:length(time_to_particles), [convert(Float32, meann(t)) for t=1:length(time_to_particles)]; color, linestyle=:dash)
            # else
            CairoMakie.lines!(ax, 1:length(time_to_particles), [convert(Float32, meann(t)) for t=1:length(time_to_particles)]; color)
            # end
        end
    end
end

function make_particle_figure(smcp310_returns, ax)
    smcp310_traces = resample.(map(first , smcp310_returns))
    #f = Figure(resolution=(500, 150))
    #ax = Axis(f[1,1])
    visualize_trace!(ax, smcp310_traces[1][2])
    return f
end

# f = make_particle_figure(smcp310_returns)


function make_mean_tracking_figure(smcp310_returns, gibbs_returns, ax)
    #f = Figure(resolution=(500, 150))
    #ax = Axis(f[1,1], title="Expected number of clusters")
    ax.xticksvisible=false
    ax.xticklabelsvisible=false
    
    plot_many_mean_n_clusters!(ax, [
            map(last, smcp310_returns),
            map(last, gibbs_returns)
        ], [:black, :red])
    return f
end

function make_log_marginal_figure(smcp310_returns, gibbs_returns, ax)
    #f = Figure(resolution=(500, 150))
    #ax = Axis(f[1,1])
    logz_gibbs = get_logz_estimates_over_time(gibbs_returns);
    logz_smcp3 = get_logz_estimates_over_time(smcp310_returns);
    gibbs_line = CairoMakie.lines!(ax, 1:length(logz_gibbs), convert(Vector{Float32}, logz_gibbs), color=:red)
    smcp3_line = CairoMakie.lines!(ax, 1:length(logz_smcp3), convert(Vector{Float32}, logz_smcp3), color=:black)
   # axislegend(ax,
    #    [smcp3_line, gibbs_line], ["SMCP3", "Loc.-Opt. SMC"],
     #   position=:rt, labelsize=12, nbanks=2
    #)
    style_ax!(ax, length(logz_gibbs))
    ax.xticksvisible = true
    ax.xticklabelsvisible = true
    return ax, smcp3_line, gibbs_line
end

# save("logz_mixture.pdf", make_log_marginal_figure(smcp310_returns, gibbs_returns))



# TODO: fix this so we are showign particule counts
function make_figure(smcp310_returns, gibbs_returns)
    gibbs_traces = resample.(map(first , gibbs_returns))
    smcp310_traces = resample.(map(first , smcp310_returns));

    f = Figure(resolution=(500, 150))
    ax = Axis(f[1,1], title="Expected number of clusters")
    #visualize_trace!(ax, smcp310_traces[1][2])
    plot_many_mean_n_clusters!(ax, [
            map(last, smcp310_returns),
            map(last, gibbs_returns)
        ], [:black, :red])
    ax.xticklabelsvisible=false
    ax.xticksvisible=false
    # smcp3_tr_ax = Axis(f[1, 1], title="Example Particle from SMCP3")
    # gibbs_tr_ax = Axis(f[2, 1], title="Example Particle from\nLocally Optimal Single-Datapoint SMC")
    # visualize_trace!(smcp3_tr_ax, smcp310_traces[1][2])
    # visualize_trace!(gibbs_tr_ax, gibbs_traces[2][2])

    # num_ax = Axis(f[3, 1], title="Inferred Number of Clusters")
    # plot_many_mean_n_clusters!(num_ax, [
    #     map(last, smcp310_returns),
    #     map(last, gibbs_returns)
    # ], [:black, :red])

    # score_ax = Axis(f[4, 1], title="Mean Log Marginal Data Likelihood Estimate", xlabel="Number of incorporated datapoints")
    # logz_gibbs = get_logz_estimates_over_time(gibbs_returns);
    # logz_smcp3 = get_logz_estimates_over_time(smcp310_returns);
    # gibbs_line = CairoMakie.lines!(score_ax, 1:length(logz_gibbs), convert(Vector{Float32}, logz_gibbs), color=:red)
    # smcp3_line = CairoMakie.lines!(score_ax, 1:length(logz_smcp3), convert(Vector{Float32}, logz_smcp3), color=:black)
    # # smcp3_line = CairoMakie.lines!(score_ax, 1:length(logz_smcp3), convert(Vector{Float32}, [x/i for (i, x) in enumerate(logz_smcp3)]), color=:black)
    # # gibbs_line = CairoMakie.lines!(score_ax, 1:length(logz_gibbs), convert(Vector{Float32}, [x/i for (i, x) in enumerate(logz_gibbs)]), color=:red)

    # axislegend(score_ax,
    #     [smcp3_line, gibbs_line], ["SMCP3", "Locally-Optimal\n1-Datapoint SMC"],
    #     position=:rt, labelsize=12
    # )

    # style_ax!(num_ax, length(logz_gibbs))
    # style_ax!(score_ax, length(logz_gibbs))
    # score_ax.xticksvisible = true
    # score_ax.xticklabelsvisible = true
    # for ax in [smcp3_tr_ax, gibbs_tr_ax, num_ax]
    #     ax.xticksvisible = true
    # end
    # rowsize!(f.layout, 4, Relative(0.35))

    return f
end


# f = make_figure(smcp310_returns, gibbs_returns)

# save("tracked_mean_mixture.pdf", f)



function combined_plot_mixture(smcp310_returns, gibbs_returns)

    f = Figure(resolution=(500, 500))
    M = 12
    MKR = :circle
    viz_ax = Axis(f[1, 1], ylabel="yₜ")
    mean_ax = Axis(f[2, 1], ylabel="E[# clusters]")
    lml_ax = Axis(f[3, 1], ylabel="log p(y₁..ₜ)", xlabel="datapoint number t")
    _ = make_particle_figure(smcp310_returns, viz_ax)
    make_mean_tracking_figure(smcp310_returns, gibbs_returns, mean_ax)
    _, blackline, redline = make_log_marginal_figure(smcp310_returns, gibbs_returns, lml_ax)
    l = Legend(f[4, 1], [#obs, 
        #bnd, 
        (
            MarkerElement(color = colors()[i], marker = markers()[i], markersize=M) for i=1:4
        )...,
        # MarkerElement(color = RGBA(0, 0, 0, 0.8), marker = markers[1], markersize=M), 
        # MarkerElement(color = RGBA(0.5, 0.5, 0.0, 0.8), marker = markers[2], markersize=M),
        # MarkerElement(color = RGBA(0.5, 0.0, 0.5, 0.8), marker = markers[3], markersize=M),
        # MarkerElement(color = RGBA(0.0, 0.5, 0.5, 0.8), marker = :hexagon, markersize=M),
        blackline, #MarkerElement(color = RGBA(0, 0, 0, 0.8), marker = :circle, markersize=M),  
        redline, #MarkerElement(color = RGBA(1, 0, 0, 0.8), marker = :circle, markersize=M), 
        #blueline], #MarkerElement(color = RGBA(0, 0, 1, 0.8), marker = :circle, markersize=M)],
        ],
        [#"Noisy observed\ntrajectory", 
         #"3σ window of\nexact posterior", 
         "Cluster 1\n(inferred)",
         "Cluster 2\n(inferred)",
         "Cluster 3\n(inferred)",
         "Cluster 4\n(inferred)",
         "SMCP3", 
         "Locally optimal\nSMC", 
         #"Resample-\nMove SMC"
         ],
        labelsize=16,
        orientation=:horizontal,
        nbanks=2
    )
    # l.tellwidth= false
    # l.tellheight = true
    f

end

f = combined_plot_mixture(smcp310_returns, gibbs_returns)
save("mixture_combined.pdf", f)