using Gen
using CairoMakie

X_STD() = 1.
Y_STD() = 1.
STEP_SIZE() = 0.3
DIM() = 100
NUM_STEPS() = 10

# Generate a fake dataset
dataset = [randn(DIM())]

# The specific dataset we randomly generated and used to make the results in the paper
# is in the file random_100d_dataset.jl
# include("random_100d_dataset.jl"); dataset = dataset_from_paper

for i in 2:NUM_STEPS()
    push!(dataset, last(dataset) + randn(DIM()))
end
for i in 1:NUM_STEPS()
    dataset[i] .+= randn(DIM()) * Y_STD()
end

### Model ###
@gen (grad) function initial_model()::Vector{Float64}
    xₜ ~ broadcasted_normal(zeros(DIM()), X_STD())
    yₜ ~ broadcasted_normal(xₜ, Y_STD())
    return xₜ
end
@gen (grad) function step_model(t, (grad)(xₜ₋₁))::Vector{Float64}
    xₜ ~ broadcasted_normal(xₜ₋₁, X_STD())
    yₜ ~ broadcasted_normal(xₜ, Y_STD())
    return xₜ
end
@gen function model(T)
    x₀ = {:init} ~ initial_model()
    steps ~ Unfold(step_model)(T, x₀)
end

### SMCP3 proposals ###

function init_proposal_gradient(x, y)
    # Compute gradient of logpdf of initial model
    prior_grad = logpdf_grad(broadcasted_normal, x, zeros(DIM()), X_STD())[1]
    obs_grad = logpdf_grad(broadcasted_normal, y, x, Y_STD())[2]
    return prior_grad .+ obs_grad
end
@gen function initial_proposal(y)::Vector{Float64}
    x_start = broadcasted_normal(zeros(DIM()), X_STD())
    x_kicked = x_start + STEP_SIZE() .* init_proposal_gradient(x_start, y)
    xₜ = {:init => :xₜ} ~ broadcasted_normal(x_kicked, sqrt(2 * STEP_SIZE()))
    return xₜ
end

function step_proposal_gradient(x_prev, x_cur, y)
    # Compute gradient of logpdf of step model
    prior_grad = logpdf_grad(broadcasted_normal, x_cur, x_prev, X_STD())[1]
    obs_grad = logpdf_grad(broadcasted_normal, y, x_cur, Y_STD())[2]
    return prior_grad .+ obs_grad
end
@gen function step_proposal(tr, y)::Vector{Float64}
    t = get_args(tr)[1] + 1
    x_prev = tr[last_x_addr(tr)]

    x_start = broadcasted_normal(x_prev, X_STD())
    x_kicked = x_start + STEP_SIZE() .* step_proposal_gradient(x_prev, x_start, y)
    xₜ = {:steps => t => :xₜ} ~ broadcasted_normal(x_kicked, sqrt(2 * STEP_SIZE()))
    return xₜ
end

# Implementing the particle filter

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

function run_logz_experiment(ys, n_particles)
    prior_results = @timed smc(ys, n_particles[1])
    resamplemove_results = @timed smc(ys, n_particles[2], mcmc_step=my_mala)
    ismc_results = @timed smc(ys, n_particles[3], initial_proposal=initial_proposal, step_proposal=step_proposal)

    return map(x -> (get_logz_estimate(x.value), x.time), (prior_results, resamplemove_results, ismc_results))
end
using StatsBase
function average_logz_experiments(y, n_particles, n_iterates)
    vals = Any[nothing for _=1:n_iterates]
    Threads.@threads for i=1:n_iterates
        vals[i] = run_logz_experiment(y, n_particles)
    end
    return [(mean(map(x -> x[n][1], vals)), std(map(x -> x[n][1], vals)), mean(map(x->x[n][2], vals))) for n=1:length(first(vals))]
end
mean(xs) = sum(xs)/length(xs)
function run_print_avg_logz_experiments(y, n_particles, n_iterates)
    (prior, rm, ismc) = average_logz_experiments(y, n_particles, n_iterates)
    println("""
    Higher estimates suggests better inference.
    y₁..ₜ = $y ; Algorithms run with $n_particles particles.
    Bootstrap particle filter: log P(y₁..ₜ) ≈ $prior
    Resample-move SMC        : log P(y₁..ₜ) ≈ $rm
    Involutive SMC           : log P(y₁..ₜ) ≈ $ismc
    """)
end

### Particle filter ###
function smc(ys, n_particles; initial_proposal=nothing, step_proposal=nothing, mcmc_step=nothing, get_intermediaries=false, ess_factor=0.2)
    initial_proposal_args(y) = isnothing(initial_proposal) ? () : ((initial_proposal, (y,)))
    step_proposal_args(y) = isnothing(step_proposal) ? () : ((step_proposal, (y,)))
    y0, yrest = Iterators.peel(ys)

    state = initialize_particle_filter(model, (0,),
        choicemap((:init => :yₜ, y0)),
        initial_proposal_args(y0)...,
        n_particles
    )

    maybe_resample_tracking_logz!(state, n_particles * ess_factor)
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
        maybe_resample_tracking_logz!(state, n_particles * ess_factor)
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

@gen function my_mala_move(tr)
    t = T(tr)
    if t == 0
        return tr
    end
    x = tr[:steps => t => :xₜ]
    y = tr[:steps => t => :yₜ]
    if t > 1
        x_prev = tr[:steps => (t-1) => :xₜ]
        x_kicked = x .+ STEP_SIZE() .* step_proposal_gradient(x_prev, x, y)
    else
        x_kicked = x .+ STEP_SIZE() .* init_proposal_gradient(x, y)
    end
    xₜ = {:steps => t => :xₜ} ~ broadcasted_normal(x_kicked, sqrt(2 * STEP_SIZE()))
    return xₜ
end

function my_mala(tr)
    tr, = Gen.mh(tr, my_mala_move, ())
    return tr
end

function make_logz_plot(results, nparticles_range)
    f = Figure(resolution=(400,400), fontsize=15)
    ax = Axis(f[1, 1])
    ax.xlabel = "Number of SMC Particles"
    ax.ylabel = "Average estimate of log P(y₁..ₜ)"
    ax.xgridvisible = false
    ax.ygridvisible = false

    lines!(nparticles_range, results[1], label="Bootstrap Particle Filter", color=:red2)
    lines!(nparticles_range, results[2], label="Resample-Move SMC", color=:blue4)
    lines!(nparticles_range, results[3], label="SMCP3", color=:black)

    axislegend(ax, position=:rb)

    f
end

function make_logz_plot_timed(results, times, CUTOFF)
    f = Figure(resolution=(400,400), fontsize=15)
    ax = Axis(f[1, 1])
    ax.xlabel = "Time (sec)"
    ax.ylabel = "Average estimate of log P(y₁..ₜ)"
    ax.xgridvisible = false
    ax.ygridvisible = false

    #println(times[1])
    lines!(filter(x -> x < CUTOFF, times[1]), results[1][1:length(filter(x -> x < CUTOFF, times[1]))], label="Bootstrap Particle Filter", color=:red2)
    lines!(filter(x -> x < CUTOFF, times[2]), results[2][1:length(filter(x -> x < CUTOFF, times[2]))], label="Resample-Move SMC", color=:blue4)
    lines!(filter(x -> x < CUTOFF, times[3]), results[3][1:length(filter(x -> x < CUTOFF, times[3]))], label="SMCP3", color=:black)

    axislegend(ax, position=:rb)

    f
end

run_print_avg_logz_experiments(dataset, [2000,800,1600], 20)

results = Dict()
PARTICLE_VALS = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 700, 1000, 1500, 2000]
for n_particles in PARTICLE_VALS
    println("n_particles = $n_particles")
    results[n_particles] = average_logz_experiments(dataset, [n_particles, n_particles, n_particles], 10)
end

my_results = [average_logz_experiments(dataset, [50, 50, 50], 10) for _ in 1:50]
my_results1d = [average_logz_experiments(dataset, [5, 5, 5], 10) for _ in 1:50]

for j in 1:3
    println((x -> (mean(x), std(x)))([my_results1d[i][j][1] for i=1:50]))
end

results[1000] = average_logz_experiments(dataset, [1000, 1000, 1000], 10)
results[1300] = average_logz_experiments(dataset, [1300, 1300, 1300], 10)
results[1700] = average_logz_experiments(dataset, [1700, 1700, 1700], 10)

nparticles_range = PARTICLE_VALS
# nparticles_range = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 700, 1000, 1300, 1500, 1700, 2000]
times_lists = [[results[i][j][3] for i in nparticles_range] for j in 1:3]
results_lists = [[results[i][j][1] for i in nparticles_range] for j in 1:3]
save("time_normalized_plot.pdf", make_logz_plot_timed(results_lists, times_lists, 1.0))
# save("particlecount_normalized_plot.pdf", make_logz_plot(results_lists, nparticles_range))

