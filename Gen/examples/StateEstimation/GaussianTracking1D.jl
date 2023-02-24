using Revise
using Gen
using GenTraceKernelDSL
DFD = GenTraceKernelDSL.DFD

X_STD() = 1.
Y_STD() = 1.
STEP_SIZE() = 0.3

### Model ###
@gen function step_model(t, xₜ₋₁)::Float64
    xₜ ~ normal(xₜ₋₁, X_STD())
    yₜ ~ normal(xₜ, Y_STD())
    return xₜ
end
@gen function model(T)
    steps ~ Unfold(step_model)(T, 0.)
end

### Helper trace-accessor functions ###
T(tr) = get_args(tr)[1]
get_xy(tr, t) =
    t == 0 ? (0. , 0.) : (tr[:steps => t => :xₜ], tr[:steps => t => :yₜ])
last_xy(tr) = get_xy(tr, T(tr))
get_ys(tr) = [get_xy(tr, t)[2] for t=0:T(tr)]
last_x_addr(tr::Gen.Trace) = last_x_addr(T(tr))
last_x_addr(t::Int) = :steps => t => :xₜ
prev_x(tr) = T(tr) < 2 ? 0. : tr[:steps => (T(tr) - 1) => :xₜ]
last_x(tr) = T(tr) == 0 ? 0. : tr[:steps => T(tr) => :xₜ]

### SMCP³ ###
function ula_mean(xₜ₋₁, xₜ, yₜ)
    grad = logpdf_grad(normal, xₜ, xₜ₋₁, X_STD())[1] + logpdf_grad(normal, yₜ, xₜ, Y_STD())[2]
    return xₜ + STEP_SIZE() * grad
end
@dist ULA(xₜ₋₁, xₜ, yₜ) = normal(ula_mean(xₜ₋₁, xₜ, yₜ), sqrt(2 * STEP_SIZE()))
@kernel function K(tr, yₜ)
    xₜ₋₁ = last_x(tr)
    u ~ normal(xₜ₋₁, X_STD())
    x ~ ULA(xₜ₋₁, u, yₜ)
    return (choicemap((:steps => T(tr) + 1 => :xₜ, x)), choicemap((:u, u)))
end
@kernel function L(tr)
    xₜ₋₁ = prev_x(tr)
    u ~ normal(xₜ₋₁, X_STD())
    return (choicemap(), choicemap((:u, u)))
end

### MALA ###
function my_mala(tr)
    tr1 = generate(step_model, (T(tr), prev_x(tr),), get_submap(get_choices(tr), :steps => T(tr)))[1]
    tr_updated, _ = Gen.mala(tr1, Gen.select(:xₜ), STEP_SIZE())
    new_tr, _, _, _ = Gen.update(tr, choicemap((last_x_addr(tr), tr_updated[:xₜ])))
    return new_tr
end

### Particle filter ###
function smc(ys, n_particles;
    step_args=(y -> ()), # args after `observation` for particle_filter_step!
    mcmc_step=nothing,
    get_intermediaries=false
)
    # Deterministic initialization
    state = initialize_particle_filter(model, (0,), choicemap(), n_particles)

    for (t, y) in enumerate(ys)
        particle_filter_step!(
            state, (t,), (UnknownChange(),),
            choicemap((:steps => t => :yₜ, y)),
            step_args(y)...
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
smcp3_pf(ys, n_particles) = 
    smc(ys, n_particles, step_args=(y -> (K, L, (y,), ())))

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

### Experiment: Bias in Log P(y₁..ₜ) Estimate
function run_logz_experiment(ys, n_particles)
    prior_results = smc(ys, n_particles)
    resamplemove_results = smc(ys, n_particles, mcmc_step=my_mala)
    smcp3_results = smcp3_pf(ys, n_particles)

    return map(get_logz_estimate, (prior_results, resamplemove_results, smcp3_results))
end
function run_logz_experiment_for_one_algo(ys, n_particles, algo)
    return get_logz_estimate(algo(ys, n_particles))
end
function average_logz_experiments(y, n_particles, n_iterates)
    # vals = Any[nothing for _=1:n_iterates]
    # Threads.@threads 
    results = []
    for (label, algo, n) in zip(
        ("prior", "resample-move", "smcp3"),
        (smc, (ys, n_p) -> smc(ys, n_p, mcmc_step=my_mala), smcp3_pf),
        n_particles
    )
        println("Time for $label: ")
        @time (res = [run_logz_experiment_for_one_algo(y, n, algo) for _=1:n_iterates])
        push!(results, res)
        # for i=1:n_iterates
        #     vals[i] = run_logz_experiment(y, n_particles)
        # end
    end
    return map(mean, results)
    # return [mean(map(x -> x[n], vals)) for n=1:length(first(vals))]
end
mean(xs) = sum(xs)/length(xs)
function run_print_avg_logz_experiments(y, n_particles, n_iterates)
    (prior, rm, smcp3) = average_logz_experiments(y, n_particles, n_iterates)
    println("""
    Higher estimates suggests better inference.
    y₁..ₜ = $y ; Algorithms run with $n_particles particles.
    Bootstrap particle filter: log P(y₁..ₜ) ≈ $prior
    Resample-move SMC        : log P(y₁..ₜ) ≈ $rm
    SMCP³                    : log P(y₁..ₜ) ≈ $smcp3
    """)
end

### Run the experiment. ###

tr = simulate(model, (100,));
run_print_avg_logz_experiments(get_retval(tr), [12, 1, 12], 10);