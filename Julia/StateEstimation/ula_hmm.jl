using Gen

X_STD() = 1.
Y_STD() = 1.
STEP_SIZE() = 0.3

@gen function initial_model()
    xₜ ~ normal(0, X_STD())
    yₜ ~ normal(xₜ, Y_STD())
end
@gen function step_model(xₜ₋₁)
    xₜ ~ normal(xₜ₋₁, X_STD())
    yₜ ~ normal(xₜ, Y_STD())
end

function smc(ys,
    generate_and_weight_initial, generate_and_weight_step,
    n_particles,
    resampling_rule
)
    yfirst, yrest = Iterators.peel(ys)
    particles = [generate_and_weight_initial(yfirst) for _=1:n_particles]

    for y in yrest
        resampled_weighted_particles = resampling_rule(particles)
        particles = [
            let (new_particle, log_inc_weight) = generate_and_weight_step(particle, y)
                (new_particle, logwt + log_inc_weight)
            end
            for (particle, logwt) in resampled_weighted_particles
        ]
    end

    return particles
end

# function prior_generate_and_weight_initial(y)
#     tr, lwt = Gen.generate(initial_model, (), choicemap((:yₜ, y)))
#     return (tr[:xₜ], lwt)
# end
# function prior_generate_and_weight_step(xₜ₋₁, y)
#     tr, lwt = Gen.generate(step_model, (xₜ₋₁,), choicemap((:yₜ, y)))
#     return (tr[:xₜ], lwt)
# end
function prior_generate_and_weight_initial(y)
    x = normal(0, X_STD())
    lw = logpdf(normal, y, x, Y_STD())
    return (x, lw)
end
function prior_generate_and_weight_step(xₜ₋₁, y)
    x = normal(xₜ₋₁, X_STD())
    lw = logpdf(normal, y, x, Y_STD())
    return (x, lw)
end

function smcp3_generate_and_weight_initial(y)
    (nextx, p_k) = simulate_smcp3(0, y)
    log_model_score = logpdf(normal, nextx, 0, X_STD()) + logpdf(normal, y, nextx, Y_STD())
    return (nextx, log_model_score - p_k)
end
function smcp3_generate_and_weight_step(xₜ₋₁, y)
    (nextx, p_k) = simulate_smcp3(xₜ₋₁, y)
    log_model_score = logpdf(normal, nextx, xₜ₋₁, X_STD()) + logpdf(normal, y, nextx, Y_STD())
    return (nextx, log_model_score - p_k)
end

# function smcp3_generate_and_weight_initial(y)
#     (nextx, auxx, log_joint_p) = simulate_K_kernel_initial(y)
#     log_p_auxx_given_nextxandy = assess_L_kernel_initial(y, nextx, auxx)
    
#     # log_model_score = Gen.assess(initial_model, (), choicemap((:yₜ, y), (:xₜ, nextx)))[1]
#     log_model_score = logpdf(normal, nextx, 0, X_STD()) + logpdf(normal, y, nextx, Y_STD())
#     logweight = log_p_auxx_given_nextxandy - log_joint_p + log_model_score

#     return (nextx, logweight)
# end
# function smcp3_generate_and_weight_step(xₜ₋₁, y)
#     nextx = run_ula

#     (nextx, auxx, log_joint_p) = simulate_K_kernel_step(xₜ₋₁, y)
#     log_p_auxx_given_nextxandy = assess_L_kernel_step(xₜ₋₁, y, nextx, auxx)
    
#     # log_model_score = Gen.assess(step_model, (xₜ₋₁,), choicemap((:yₜ, y), (:xₜ, nextx)))[1]
#     log_model_score = logpdf(normal, nextx, xₜ₋₁, X_STD()) + logpdf(normal, y, nextx, Y_STD())
#     logweight = log_p_auxx_given_nextxandy - log_joint_p + log_model_score

#     return (nextx, logweight)
# end

function smcp3_generate_and_weight_step_ula_L(xₜ₋₁, y)
    (nextx, auxx, log_joint_p) = simulate_K_kernel_step(xₜ₋₁, y)
    log_p_auxx_given_nextxandy = assess_ula_L_step(xₜ₋₁, y, nextx, auxx)
    assess_ula_L_step
    log_model_score = Gen.assess(step_model, (xₜ₋₁,), choicemap((:yₜ, y), (:xₜ, nextx)))[1]
    logweight = log_p_auxx_given_nextxandy - log_joint_p + log_model_score

    return (nextx, logweight)
end

function resamplemove_generate_and_weight_initial(y)
    tr, lwt = Gen.generate(initial_model, (), choicemap((:yₜ, y)))
    newtrace, _ = Gen.mala(tr, select(:xₜ), STEP_SIZE())
    return (newtrace[:xₜ], lwt)
end
function resamplemove_generate_and_weight_step(xₜ₋₁, y)
    tr, lwt = Gen.generate(step_model, (xₜ₋₁,), choicemap((:yₜ, y)))
    newtrace, _ = Gen.mala(tr, select(:xₜ), STEP_SIZE())
    return (newtrace[:xₜ], lwt)
end

vanilla_experiment(ys, n_particles) = smc(ys,
    prior_generate_and_weight_initial, prior_generate_and_weight_step,
    n_particles,
    ESSThreshold(n_particles / 4)
)
smcp3_experiment(ys, n_particles) = smc(ys,
    smcp3_generate_and_weight_initial, smcp3_generate_and_weight_step,
    n_particles,
    ESSThreshold(n_particles / 4)
)
rm_experiment(ys, n_particles) = smc(ys,
    resamplemove_generate_and_weight_initial, resamplemove_generate_and_weight_step,
    n_particles,
    ESSThreshold(n_particles / 4)
)
to_logsumexp_experiment(experiment) = (y, n) -> logsumexp(map(last, experiment(y, n))) - log(n)

function average_timed_experiments(y, n_particles, n_iterates)
    means = []
    for (label, experiment, n) in zip(
        ("prior", "prior, v2", "smcp3"),
        map(to_logsumexp_experiment, (vanilla_experiment, vanilla_experiment, smcp3_experiment)),
        n_particles
    )
        GC.gc()
        println("Time for $label: ")
        @time(vals = [experiment(y, n) for _=1:n_iterates])
        push!(means, mean(vals))
    end
    return means
end

# function run_experiment(ys, n_particles)
#     prior_smc_samples = smc(ys,
#         prior_generate_and_weight_initial, prior_generate_and_weight_step,
#         n_particles,
#         ESSThreshold(n_particles / 4)
#     )
#     smcp3_samples = smc(ys,
#         smcp3_generate_and_weight_initial, smcp3_generate_and_weight_step,
#         n_particles,
#         ESSThreshold(n_particles / 4)
#     )
#     resamplemove_samples = smc(ys,
#         resamplemove_generate_and_weight_initial, resamplemove_generate_and_weight_step,
#         n_particles,
#         ESSThreshold(n_particles / 4)
#     )
    
#     prior_logweight = logsumexp(map(last, prior_smc_samples)) - log(n_particles)
#     smcp3_logweight = logsumexp(map(last, smcp3_samples)) - log(n_particles)
#     rm_logweight = logsumexp(map(last, resamplemove_samples)) - log(n_particles)

#     return (prior_logweight, rm_logweight, smcp3_logweight)
# end
# function average_experiments(y, n_particles, n_iterates)
#     vals = [run_experiment(y, n_particles) for _=1:n_iterates]
#     return [mean(map(x -> x[n], vals)) for n=1:length(first(vals))]
# end
# function run_print_avg_experiments(y, n_particles, n_iterates)
#     (prior, rm, smcp3) = average_experiments(y, n_particles, n_iterates)
#     println("""
#     Higher estimates suggests better inference.
#     y₁..ₜ = $y ; Algorithms run with $n_particles particles.
#     Bootstrap particle filter: log P(y₁..ₜ) ≈ $prior
#     Resample-move SMC        : log P(y₁..ₜ) ≈ $rm
#     SMCP3                    : log P(y₁..ₜ) ≈ $smcp3
#     """)
# end
mean(xs) = sum(xs)/length(xs)

# set each particle weight to the average weight
ESSThreshold(threshold) =
    (function maybe_resample(particles)
        log_weights = map(last, particles)
        (log_sum_weights, log_normalized_weights) = Gen.normalize_weights(log_weights)
        log_mean_weight = log_sum_weights - log(length(particles))
        ess = Gen.effective_sample_size(log_normalized_weights)
        if ess < threshold
            probs = exp.(log_normalized_weights)
            indices = [categorical(probs) for _=1:length(particles)]
            return [(particles[i][1], log_mean_weight) for i in indices]
        else
            return particles
        end
    end)

######## SMCP3 Kernels ###########

# Optimized function that computes samples from the K kernel and computes the K/L weight
# in one go
function simulate_smcp3(mean, y)
    auxx = normal(mean, X_STD())
    nextx, log_p_next_x = run_ula(mean, auxx, y)

    return (nextx, log_p_next_x)
end

function simulate_K_kernel(mean, y)
    auxx = normal(mean, X_STD())
    log_p_aux = logpdf(normal, auxx, mean, X_STD())

    nextx, log_p_next_x = run_ula(mean, auxx, y)

    log_joint_p = log_p_aux + log_p_next_x

    return (nextx, auxx, log_joint_p)
end
simulate_K_kernel_initial(y) = simulate_K_kernel(0, y)
simulate_K_kernel_step(xₜ₋₁, y) = simulate_K_kernel(xₜ₋₁, y)

function assess_L_kernel_step(xₜ₋₁, y, nextx, auxx)
    return logpdf(normal, auxx, xₜ₋₁, X_STD())
end
function assess_L_kernel_initial(y, nextx, auxx)
    return logpdf(normal, auxx, 0, X_STD())
end

function L_ula_params(prior_mean, x_postkick)
    std = sqrt(2 * STEP_SIZE())
    grad = logpdf_grad(normal, x_postkick, prior_mean, 1)[1]
    mean = x_postkick + grad*STEP_SIZE()
    return (mean, std)
end
function assess_ula_L_step(xₜ₋₁, y, nextx, auxx)
    (mean, std) = L_ula_params(xₜ₋₁, nextx)
    return logpdf(normal, auxx, mean, std)
end

function ula_params(prior_mean, x_prekick, y)
    std = sqrt(2 * STEP_SIZE())

    prior_grad = logpdf_grad(normal, x_prekick, prior_mean, 1)[1]
    obs_grad = logpdf_grad(normal, y, x_prekick, 1)[2]
    grad = obs_grad + prior_grad

    mean = x_prekick + grad*STEP_SIZE()

    return (mean, std)
end
function run_ula(prior_mean, x_prekick, y)
    (mean, std) = ula_params(prior_mean, x_prekick, y)
    x_postkick = normal(mean, std)
    log_p_jump = logpdf(normal, x_postkick, mean, std)

    return (x_postkick, log_p_jump)
end

function run_print_avg_experiments(y, n_particles, n_iterates)
    (prior, rm, smcp3) = average_timed_experiments(y, n_particles, n_iterates)
    println("""
    Higher estimates suggests better inference.
    Algorithms run with $n_particles particles.
    Bootstrap particle filter: log P(y₁..ₜ) ≈ $prior
    Bootstrap, again        : log P(y₁..ₜ) ≈ $rm
    SMCP3                    : log P(y₁..ₜ) ≈ $smcp3
    """)
end
run_print_avg_experiments(get_retval(tr), [12, 1, 12], 1000);