using Distributions: Gamma, Normal, Categorical
using Plots, StatsPlots
using Gen: logsumexp
using CSV, DataFrames
import StatsBase

include("../shared.jl")
include("../dpmm-data-structures.jl")
include("gaussian-clusters.jl")
include("../locally-optimal-smc.jl")
include("../smcp3.jl")
include("gaussian-visualization.jl")
include("utils.jl")
include("../sm_mcmc.jl")

# Hyperparameters
HYPERS = GaussianHyperparameters(0, 1 / 100, 1 / 2, 1 / 2)
ALPHA = 1.0

### Galaxy data ###
N = 100
REPS = 100

# Sorted
using Serialization
galaxies_data = map(Float64, CSV.read("Julia/MixtureModel/datasets/galaxies.csv", DataFrame)[!, 1])

N_v = 1200
v_to_s = 0.19/7.2 / 3 #* 1.1
v_to_r = 0.19/5.89 # * 1.29 * 1.7
REPS = 30
N_splits = 10
galaxies_data = sort(galaxies_data, rev=true)
N_s = Int(floor(N_v * v_to_s))
N_r = Int(floor(N_v * v_to_r))
println("Runtimes for sorted run ($REPS reps); (vanilla, SMCP3, resample-move) =")
GC.gc()
@time (gibbs_returns = [run_gibbs_smc(galaxies_data, N_v, HYPERS, GaussianCluster, ALPHA, LocallyOptimalSMCOptions(10, 0, N_v/5)) for _=1:REPS];);
GC.gc()
@time (smcp3_returns = [run_smcp3(galaxies_data, N_s, HYPERS, GaussianCluster, ALPHA, SMCP3Options(N_s/5, N_splits)) for _ in 1:REPS];);
GC.gc()
@time (rm_returns = [run_gibbs_smc(galaxies_data, N_r, HYPERS, GaussianCluster, ALPHA, LocallyOptimalSMCOptions(1, 1, N_r/5); rejuv_sweep=sm_mcmc!) for _=1:REPS]);
gibbs_traces = map(first ∘ first , gibbs_returns);
smcp3_traces = map(first ∘ first , smcp3_returns);
rm_traces = map(first ∘ first , rm_returns);
gibbs_smc_results = [logmeanexp(res[1][2]) for res in gibbs_returns]
smcp3_results = [logmeanexp(res[1][2]) for res in smcp3_returns]
rm_smc_results = [logmeanexp(res[1][2]) for res in rm_returns]
println("Locally Optimal SMC results (galaxies, sorted): $(StatsBase.mean(gibbs_smc_results)) +/- $(StatsBase.std(gibbs_smc_results))")
println("SMCP3 results (galaxies, sorted): $(StatsBase.mean(smcp3_results)) +/- $(StatsBase.std(smcp3_results))")
println("RM SMC results (galaxies, sorted): $(StatsBase.mean(rm_smc_results)) +/- $(StatsBase.std(rm_smc_results))")

# Unsorted
# N_v = 500
# N_s = Int(floor(N_v * v_to_s))
# N_r = Int(floor(N_v * v_to_r))
galaxies_data = shuffle(galaxies_data)
println("Runtimes for unsorted run ($REPS reps); (vanilla, SMCP3, resample-move) =")
GC.gc()
@time (gibbs_returns = [run_gibbs_smc(galaxies_data, N_v, HYPERS, GaussianCluster, ALPHA, LocallyOptimalSMCOptions(10, 0, N_v/5)) for _=1:REPS];);
GC.gc()
@time (smcp3_returns = [run_smcp3(galaxies_data, N_s, HYPERS, GaussianCluster, ALPHA, SMCP3Options(N_s/5, N_splits)) for _ in 1:REPS];);
GC.gc()
@time (rm_returns = [run_gibbs_smc(galaxies_data, N_r, HYPERS, GaussianCluster, ALPHA, LocallyOptimalSMCOptions(1, 1, N_r/5); rejuv_sweep=sm_mcmc!) for _=1:REPS]);
gibbs_traces = map(first ∘ first , gibbs_returns)
smcp3_traces = map(first ∘ first , smcp3_returns);
rm_traces = map(first ∘ first , rm_returns)
gibbs_smc_results = [logmeanexp(res[1][2]) for res in gibbs_returns]
smcp3_results = [logmeanexp(res[1][2]) for res in smcp3_returns]
rm_smc_results = [logmeanexp(res[1][2]) for res in rm_returns]
println("Locally Optimal SMC results (galaxies, unsorted): $(StatsBase.mean(gibbs_smc_results)) +/- $(StatsBase.std(gibbs_smc_results))")
println("SMCP3 results (galaxies, unsorted): $(StatsBase.mean(smcp3_results)) +/- $(StatsBase.std(smcp3_results))")
println("RM SMC results (galaxies, unsorted): $(StatsBase.mean(gibbs_smc_results)) +/- $(StatsBase.std(gibbs_smc_results))")

println("Particle counts: (vanilla, SMCP3, resample_move) = $((N_v, N_s, N_r))")