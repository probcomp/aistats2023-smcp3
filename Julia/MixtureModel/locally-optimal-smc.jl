using Random: shuffle
include("gibbs.jl")

function locally_optimal_smc_step!(trace, i; retained=nothing, get_proposal_probability=false)
    # Choose a cluster to reincorporate it into
    all_cluster_ids = collect(keys(trace.clusters))

    if isnothing(retained) || haskey(trace.clusters, retained.assignments[i])
        new_cluster_id = gensym()
    else
        new_cluster_id = retained.assignments[i]
    end
    new_cluster = singleton_cluster(trace, i)
    push!(all_cluster_ids, new_cluster_id)

    log_priors      = [crp_log_prior_predictive(trace, cluster_id, 1) for cluster_id in all_cluster_ids]
    log_likelihoods = [conditional_likelihood(trace, cluster_id, new_cluster) for cluster_id in all_cluster_ids]
    log_joints      = log_priors .+ log_likelihoods
    log_total       = logsumexp(log_joints)
    cluster_probabilities = exp.(log_joints .- log_total)
    if isnothing(retained)
        chosen_cluster_id = all_cluster_ids[rand(Categorical(cluster_probabilities))]
    else
        chosen_cluster_id = retained.assignments[i]
    end
    
    incorporate_point!(trace, i, chosen_cluster_id)
    
    if get_proposal_probability # Return the probability of this function making these choices.
        return log_joints[findfirst(all_cluster_ids .== chosen_cluster_id)] - log_total
    else # Return the importance weight from this proposal.
        # The weight is the logsumexp of the choices.
        # The reason is that the proposal is locally optimal:
        #   p(x) / q(x) = p(x) / (p(x)/sum(p(x) for x in xs)) = sum(p(x) for x in xs)
        return log_total
    end
end


struct LocallyOptimalSMCOptions
    rejuvenation_frequency :: Int
    rejuvenation_iters :: Int
    ess_threshold :: Float64
end

function run_gibbs_smc(
    data::Vector, K::Int, hypers::Hyperparameters,
    C::Type, alpha::Real, options=LocallyOptimalSMCOptions(20, 1, K/5);
    retained=nothing,
    rejuv_sweep = gibbs_sweep!,
    return_record = false
)

    K_unretained = isnothing(retained) ? K : K-1

    traces = [create_initial_dpmm_trace(hypers, C, data, alpha) for _ in 1:K]
    weights = zeros(K)

    if return_record
        trace_record, weight_record = [], []
    else
        trace_record, weight_record = nothing, nothing
    end

    for i in 1:length(data)

        # Advance
        for (j, trace) in enumerate(traces) 
            weights[j] += locally_optimal_smc_step!(trace, i; retained= j == K ? retained : nothing)

            # Move
            if options.rejuvenation_frequency > 0 && i % options.rejuvenation_frequency == 0
                if !isnothing(retained) && options.rejuvenation_iters > 0
                    @error "Code does not yet support Gibbs rejuvenation in cSMC"
                end
                for _ in 1:options.rejuvenation_iters
                    traces[j] = rejuv_sweep(trace)
                end
            end
        end

        if return_record
            push!(trace_record, map(copytrace, traces))
            push!(weight_record, copy(weights))
        end
        (traces, weights) = maybe_resample(traces, weights, options.ess_threshold; retained)
    end

    return ((traces, weights), (trace_record, weight_record))
end

function resample_unweighted_trace(traces, weights)
    normalized_weights = exp.(weights .- logsumexp(weights))
    return traces[rand(Categorical(normalized_weights))]
end