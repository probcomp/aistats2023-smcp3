include("partition_utils.jl")
include("../distributions/other_distributions.jl")
include("loc_opt.jl")
include("smart_sm.jl")

### Helpers ###
function maybe_resample_tracking_logz!(state, ess_threshold)
    (log_sum_weights, log_normalized_weights) = Gen.normalize_weights(state.log_weights)
    ess = Gen.effective_sample_size(log_normalized_weights)
    if ess < ess_threshold
        log_mean_weight = log_sum_weights - log(length(state.traces))
        probs = exp.(log_normalized_weights)
        indices = [categorical(probs) for _=1:length(state.traces)]
        state.traces = [state.traces[i] for i in indices]
        state.log_weights = [log_mean_weight for _ in indices]
    end
    return state
end
get_logz_estimate(state) = logsumexp(state.log_weights) - log(length(state.log_weights))

### Particle filter ###
function smc(dpmm, data, n_particles;
    step_args=(y -> ()), # args after `observation` for particle_filter_step!
    kwargs=(),
    mcmc_step=nothing,
    get_intermediaries=false
)
    # Deterministic initialization
    state = initialize_particle_filter(dpmm, ([],), choicemap((:condition_this, true)), n_particles)

    for t in keys(data)
        particle_filter_step!(
            state, (data[1:t],), (UnknownChange(),), EmptyChoiceMap(),
            step_args(data[1:t])...;
            kwargs...
        )
        maybe_resample_tracking_logz!(state, n_particles/5)
        if !isnothing(mcmc_step)
            state.traces = map(mcmc_step, state.traces)
        end
        if get_intermediaries
            push!(states, (copy(state.traces), copy(state.log_weights)))
        end    

    end

    if get_intermediaries
        return (state, states)
    else
        return state
    end
end

### SMC Algorithms ###
smc_locopt(dpmm, data, n_particles) = smc(dpmm, data, n_particles,
    step_args = ys -> (k_locopt, l_locopt, (ys,), ())
)
smcp3(dpmm, data, n_particles; n_split_proposals=1) = smc(dpmm, data, n_particles,
    step_args = ys -> (k_targetted_splitmerge, l_targetted_splitmerge,
                        (ys, n_split_proposals), (ys, n_split_proposals)),
    kwargs = (; check_are_inverses=true)
)