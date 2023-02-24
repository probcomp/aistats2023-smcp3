using Distributions: Gamma, Normal, Categorical
using Plots, StatsPlots
using Gen: logsumexp
using CSV, DataFrames
import StatsBase

include("../shared.jl")
include("../dpmm-data-structures.jl")
include("gaussian-clusters-fixedstd.jl")
include("../locally-optimal-smc.jl")
# include("../gibbs.jl") - TODO: compare to Gibbs?
include("../smcp3.jl")
include("gaussian-visualization.jl")
include("utils.jl")

include("../sm_mcmc.jl")

# Hyperparameters
HYPERS = FSGaussianHyperparameters(0, 10000, 1)
ALPHA = 3.0

# Create many synthetic datasets
generate_synthetic_dataset(hypers, alpha, n) = generate_synthetic_trace(hypers, alpha, n).data
generate_synthetic_trace(hypers, alpha, n; sorted=false) = generate_synthetic_fsgaussian_trace(hypers, alpha, n; sorted)

###
# Experiments
###
N = 10
REPS = 20

### Synthetic data, generated from inference prior ###

HYPERS = FSGaussianHyperparameters(0, 1_000, 1)
gt_trace = generate_synthetic_trace(HYPERS, ALPHA, 100; sorted=false)
synthetic_dataset = gt_trace.data 

gibbs_returns = [run_gibbs_smc(synthetic_dataset, N, HYPERS, FSGaussianCluster, ALPHA, LocallyOptimalSMCOptions(1, 1, N/5); rejuv_sweep=sm_mcmc!) for _=1:REPS];
smcp310_returns = [run_smcp3(synthetic_dataset, N, HYPERS, FSGaussianCluster, ALPHA, SMCP3Options(N/5, 10)) for _ in 1:REPS];
gibbs_traces = resample.(map(first , gibbs_returns))
smcp310_traces = resample.(map(first , smcp310_returns));

gibbs_smc_results = [logmeanexp(res[1][2]) for res in gibbs_returns]
smcp310_results = [logmeanexp(res[1][2]) for res in smcp310_returns]
println("Gibbs SMC results (synthetic, model-average, $N particles): $(StatsBase.mean(gibbs_smc_results)) +/- $(StatsBase.std(gibbs_smc_results))")
println("smcp3 results (synthetic, model-average, $N particles, 10 split particles): $(StatsBase.mean(smcp310_results)) +/- $(StatsBase.std(smcp310_results))")

### Synthetic data, generated narrower prior than inference ###
REPS = 2
ALPHA = 20.0
HYPERS = FSGaussianHyperparameters(0, 60, 1)
gt_trace = generate_synthetic_trace(HYPERS, ALPHA, 100; sorted=false)
synthetic_dataset = gt_trace.data 
HYPERS = FSGaussianHyperparameters(0, 100_000_000, 1)

@time gibbs_returns_rejuv = [run_gibbs_smc(synthetic_dataset, N, HYPERS, FSGaussianCluster, ALPHA, LocallyOptimalSMCOptions(1, 1, N/5); rejuv_sweep=sm_mcmc!) for _=1:REPS];
@time gibbs_returns = [run_gibbs_smc(synthetic_dataset, N, HYPERS, FSGaussianCluster, ALPHA, LocallyOptimalSMCOptions(0, 1, N/5); rejuv_sweep=sm_mcmc!) for _=1:REPS];
@time smcp310_returns = [run_smcp3(synthetic_dataset, N, HYPERS, FSGaussianCluster, ALPHA, SMCP3Options(N/5, 10)) for _ in 1:REPS];
gibbs_traces = resample.(map(first , gibbs_returns))
gibbs_traces_rejuv = resample.(map(first , gibbs_returns_rejuv))
smcp310_traces = resample.(map(first , smcp310_returns));

gibbs_smc_results = [logmeanexp(res[1][2]) for res in gibbs_returns]
gibbs_rejuv_results = [logmeanexp(res[1][2]) for res in gibbs_returns_rejuv]
smcp310_results = [logmeanexp(res[1][2]) for res in smcp310_returns]
println("Gibbs SMC results (synthetic, challenging, $N particles): $(StatsBase.mean(gibbs_smc_results)) +/- $(StatsBase.std(gibbs_smc_results))")
println("Gibbs SMC + Split/Merge MCMC results (synthetic, challenging, $N particles): $(StatsBase.mean(gibbs_rejuv_results)) +/- $(StatsBase.std(gibbs_rejuv_results))")
println("smcp3 results (synthetic, challenging, $N particles, 10 split particles): $(StatsBase.mean(smcp310_results)) +/- $(StatsBase.std(smcp310_results))")

# using Serialization

# dict = Dict(
#     :gt_trace => gt_trace,
#     :gibbs_returns_rejuv => gibbs_returns_rejuv,
#     :gibbs_returns => gibbs_returns,
#     :smcp310_returns => smcp310_returns
# )
# serialize("saves/interesing_300dp_$i.jld", dict); i += 1














# function generate_results_on_synthetic_data(N, REPS)
#     gt_trace = generate_synthetic_trace(HYPERS, ALPHA, 50; sorted=false)
#     synthetic_dataset = gt_trace.data # [1.0 + rand(Normal(0, 1)) for _=1:12]# gt_trace.data
#     return (gt_trace, generate_results(synthetic_dataset, N, REPS))
# end
# function generate_results(data, N, REPS)
#     gibbs_returns = [run_gibbs_smc(data, N, HYPERS, FSGaussianCluster, ALPHA, LocallyOptimalSMCOptions(10, 0, N/5)) for _=1:REPS];
#     smcp310_returns = [run_smcp3(data, N, HYPERS, FSGaussianCluster, ALPHA, SMCP3Options(N/5, 10)) for _ in 1:REPS];
#     # smcp31_returns = [run_smcp3(data, N, HYPERS, FSGaussianCluster, ALPHA, SMCP3Options(N/5, 10)) for _ in 1:REPS];
#     # smcp3_old_returns = [run_smcp3_old(data, N, HYPERS, FSGaussianCluster, ALPHA) for _ in 1:REPS];
#     gibbs_traces = resample.(map(first , gibbs_returns))
#     smcp310_traces = resample.(map(first , smcp310_returns));
#     # smcp31_traces = resample.(map(first , smcp31_returns));
#     smcp3_old_traces = resample.(map(first , smcp3_old_returns));
#     gibbs_smc_results = [logmeanexp(res[1][2]) for res in gibbs_returns]
#     smcp310_results = [logmeanexp(res[1][2]) for res in smcp310_returns]
#     # smcp31_results = [logmeanexp(res[1][2]) for res in smcp31_returns]
#     # smcp3_old_results = [logmeanexp(res[1][2]) for res in smcp3_old_returns]
#     # println("Gibbs SMC results (synthetic, $N particles): $(StatsBase.mean(gibbs_smc_results)) +/- $(StatsBase.std(gibbs_smc_results))")
#     # println("smcp3 results (synthetic, $N particles, 10 split particles): $(StatsBase.mean(smcp310_results)) +/- $(StatsBase.std(smcp310_results))")
#     # println("smcp3 results (synthetic, $N particles, 1 split particles): $(StatsBase.mean(smcp31_results)) +/- $(StatsBase.std(smcp31_results))")
#     # println("smcp3 results (synthetic, $N particles, 1 split particles -- OLD IMPLEMENTATION): $(StatsBase.mean(smcp3_old_results)) +/- $(StatsBase.std(smcp3_old_results))")
#     return (gibbs_smc_results, smcp310_results)
# end
# function get_sorted_results(n_datas, N, REPS)
#     traces_and_results = [generate_results_on_synthetic_data(N, REPS) for _=1:n_datas]
#     mean_gibbs_results = [StatsBase.mean(gibbsres) for (_, (gibbsres, _)) in traces_and_results]
#     mean_smcp3_results = [StatsBase.mean(smcp3res) for (_, (_, smcp3res)) in traces_and_results]
#     std_gibbs_results = [StatsBase.std(gibbsres) for (_, (gibbsres, _)) in traces_and_results]
#     std_smcp3_results = [StatsBase.std(smcp3res) for (_, (_, smcp3res)) in traces_and_results]
    
#     diffs = mean_smcp3_results .- mean_gibbs_results
#     perm = sortperm(diffs, rev=true)
#     return (
#         (mean_gibbs_results[perm], std_gibbs_results[perm]),
#         (mean_smcp3_results[perm], std_smcp3_results[perm]),
#         map(first, traces_and_results)[perm]
#     )
# end