using CSV
using DataFrames: DataFrame

include("../shared.jl")
include("../dpmm-data-structures.jl")
include("string_clusters.jl")
include("../locally-optimal-smc.jl")
include("../smcp3.jl")
include("../gibbs.jl")
include("../sm_mcmc.jl")

dataset = "hospital"
dirty_table = CSV.File("Julia/MixtureModel/datasets/$dataset.csv") |> DataFrame;

possibilities = Dict(col => remove_missing(unique(collect(dirty_table[!, col])))
                        for col in names(dirty_table))

string_dataset = dirty_table.MeasureName

HYPERS = make_hypers(unique(possibilities["MeasureName"]))
ALPHA = 1.0

REPS = 2

# Test vanilla SMC
vanilla_smc_results = []
println("\nVanilla SMC runs:")
for _ in 1:REPS
    GC.gc()
    @time ((traces, weights), _) = run_gibbs_smc([s for s in string_dataset], 32*5, HYPERS, StringCluster, ALPHA, LocallyOptimalSMCOptions(100, 1, 20));
    resampled_trace = resample_unweighted_trace(traces, weights)
    push!(vanilla_smc_results, (weight=logmeanexp(weights), n_clusters=length(resampled_trace.clusters)))
end
println("Locally-Optimal SMC: $(mean(map(x -> x.weight, vanilla_smc_results))) +/- $(StatsBase.std(map(x -> x.weight, vanilla_smc_results)))")

# Test vanilla SMC
rm_smc_results = []
println("\nResample-move SMC runs:")
for _ in 1:REPS
    @time ((traces, weights), _) = run_gibbs_smc(
        [s for s in string_dataset], 32, HYPERS, StringCluster, ALPHA, LocallyOptimalSMCOptions(1, 1, 20);
        rejuv_sweep=(sm_mcmc!)
    );
    resampled_trace = resample_unweighted_trace(traces, weights)
    push!(rm_smc_results, (weight=logmeanexp(weights), n_clusters=length(resampled_trace.clusters), example_trace=resampled_trace))
end
println("Loc-Opt SMC + MCMC Split/Merge: $(mean(map(x -> x.weight, rm_smc_results))) +/- $(StatsBase.std(map(x -> x.weight, rm_smc_results)))")

# Test smcp3
println("\nSMCP3 runs:")
smcp3_results = []
for _ in 1:REPS
    @time ((traces, weights), _) = run_smcp3([s for s in string_dataset], 2, HYPERS, StringCluster, ALPHA);
    push!(smcp3_results, (;weight=logmeanexp(weights)))
end
println("SMCP3: $(mean(map(x -> x.weight, smcp3_results))) +/- $(StatsBase.std(map(x -> x.weight, smcp3_results)))")