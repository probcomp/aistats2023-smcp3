using Gen

X_STD() = 1.
Y_STD() = 1.
STEP_SIZE() = 0.3

### Model ###
@gen (grad) function initial_model()::Float64
    xₜ ~ normal(0, X_STD())
    yₜ ~ normal(xₜ, Y_STD())
    return xₜ
end
@gen (grad) function step_model(t, (grad)(xₜ₋₁))::Float64
    xₜ ~ normal(xₜ₋₁, X_STD())
    yₜ ~ normal(xₜ, Y_STD())
    return xₜ
end
@gen function model(T)
    x₀ = {:init} ~ initial_model()
    steps ~ Unfold(step_model)(T, x₀)
end

### Helper trace-accessor functions ###
T(tr) = get_args(tr)[1]
get_xy(tr, t) =
    if t == 0
        (tr[:init => :xₜ], tr[:init => :yₜ])
    else
        (tr[:steps => t => :xₜ], tr[:steps => t => :yₜ])
    end
last_xy(tr) = get_xy(tr, T(tr))
get_ys(tr) = [get_xy(tr, t)[2] for t=0:T(tr)]
last_x_addr(tr) =
    if T(tr) == 0
        :init => :xₜ
    else
        :steps => T(tr) => :xₜ
    end
prev_x(tr) =
    if T(tr) == 1
        tr[:init => :xₜ]
    else
        tr[:steps => (T(tr) - 1) => :xₜ]
    end

### SMCP3 Transition ###
#=
The SMCP3 algorithm used here has the property that we can actually
implement an equivalent SMCP3 transition by simply using a proposal
distribution which has some untraced randomness in it.
(SMCP3 provides the justification that this is still a valid
inference algorithm.)
For now, I am using this trick to implement the SMCP3 algorithm even though
we have not finished the SMCP3 Gen DSL yet.
Note that I run ULA using gradients on artificial traces constructed for
`initial_model` and `step_model` because there appear to be bugs in
the methods for computing gradients for the Unfold combinator and the static DSL
(or I am somehow misunderstanding how to use the gradient methods).
=#
function ula_mean(tr)
    (_, _, grads) = Gen.choice_gradients(tr, Gen.select(:xₜ))
    mean = tr[:xₜ] + STEP_SIZE() * grads[:xₜ]
    return mean
end
@dist ULA(tr) = normal(ula_mean(tr), sqrt(2 * STEP_SIZE()))
@gen function initial_smcp3_ula_proposal(y₀)
    tr1 = generate(initial_model, (), choicemap((:yₜ, y₀)))[1]
    {:init => :xₜ} ~ ULA(tr1)
end
@gen function step_smcp3_ula_proposal(tr, yₜ)
    t = get_args(tr)[1] + 1
    xₜ₋₁ = tr[last_x_addr(tr)]
    tr1 = generate(step_model, (t, xₜ₋₁), choicemap((:yₜ, yₜ)))[1]
    {:steps => t => :xₜ} ~ ULA(tr1)
end

### Mala implementation to get around the challenges using the Gen gradient methods ###
#=
Again, because the gradient methods don't seem to work with the Unfold combinator,
we can't just call `Gen.mala(tr, STEP_SIZE())`; instead I need to call `Gen.mala`
on a trace for either the `initial_model` or `step_model`.
=#
function my_mala(tr)
    if T(tr) == 0
        tr1 = generate(initial_model, (), get_submap(get_choices(tr), :init))[1]
    else
        tr1 = generate(step_model, (T(tr), prev_x(tr),), get_submap(get_choices(tr), :steps => T(tr)))[1]
    end
    tr_updated, _ = Gen.mala(tr1, Gen.select(:xₜ), STEP_SIZE())
    new_tr, _, _, _ = Gen.update(tr, choicemap((last_x_addr(tr), tr_updated[:xₜ])))
    return new_tr
end

### Particle filter ###
function smc(ys, n_particles; initial_proposal=nothing, step_proposal=nothing, mcmc_step=nothing, get_intermediaries=false)
    initial_proposal_args(y) = isnothing(initial_proposal) ? () : ((initial_proposal, (y,)))
    step_proposal_args(y) = isnothing(step_proposal) ? () : ((step_proposal, (y,)))
    y0, yrest = Iterators.peel(ys)

    state = initialize_particle_filter(model, (0,),
        choicemap((:init => :yₜ, y0)),
        initial_proposal_args(y0)...,
        n_particles
    )

    maybe_resample_tracking_logz!(state, n_particles/5)
    if get_intermediaries
        states = [(copy(state.traces), copy(state.log_weights))]
    end

    if !isnothing(mcmc_step)
        state.traces = map(mcmc_step, state.traces)
    end

    for (t, y) in enumerate(yrest)
        particle_filter_step!(
            state, (t,), (UnknownChange(),),
            choicemap((:steps => t => :yₜ, y)), step_proposal_args(y)...
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

# This is like the function Gen.maybe_resample!, but when it resamples,
# rather than setting the particle weights to 1, it sets them to the average
# particle weight.  This results in the average particle weight being an 
# estimate of the marginal data likelihood, at every step.
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

### Experiment: Bias in Log P(y₁..ₜ) Estimate
function run_logz_experiment(ys, n_particles)
    prior_results = smc(ys, n_particles)
    resamplemove_results = smc(ys, n_particles, mcmc_step=my_mala)
    smcp3_results = smc(ys, n_particles, initial_proposal=initial_smcp3_ula_proposal, step_proposal=step_smcp3_ula_proposal)

    return map(get_logz_estimate, (prior_results, resamplemove_results, smcp3_results))
end
function average_logz_experiments(y, n_particles, n_iterates)
    vals = Any[nothing for _=1:n_iterates]
    Threads.@threads for i=1:n_iterates
        vals[i] = run_logz_experiment(y, n_particles)
    end
    return [mean(map(x -> x[n], vals)) for n=1:length(first(vals))]
end
mean(xs) = sum(xs)/length(xs)