include("boilerplate.jl")
include("noisy_renderer.jl")
include("inference.jl")
include("model.jl")

mean(v) = sum(v) / length(v)
name = Dict(grid_smcp3 => "smcp3", bootstrap_particle_filter => "bspf")

experiments = [
   (grid_smcp3, 2),  #=(bootstrap_particle_filter, 2),=# (bootstrap_particle_filter, Int(ceil(2* 4.49/0.5))),
   (grid_smcp3, 4),  #=(bootstrap_particle_filter, 4),=# (bootstrap_particle_filter, Int(ceil(4* 11.58/0.93))),
   (grid_smcp3, 6),  #=(bootstrap_particle_filter, 6),=# (bootstrap_particle_filter, Int(ceil(6* 17.20/1.66))),
   (grid_smcp3, 8),  #=(bootstrap_particle_filter, 8),=# (bootstrap_particle_filter, Int(ceil(8* 22.7/2.26))),
   (grid_smcp3, 10), #=(bootstrap_particle_filter, 10)]=# (bootstrap_particle_filter, Int(ceil(10 * 30.99/2.81)))]

logmeanexp(xs) = let xxs = filter(!isnan, xs); logsumexp(xxs) - log(length(xxs)) end

function avg_weight_experiment(n_iters, num_particles, algo, obs)
    res, tot_time = 0, 0
    for i=1:n_iters
        run_res, t = @timed algo(obs, num_particles)[2][end]
        println("run $i of $(name[algo]) with $num_particles particles took $t:" * string(run_res))
        res += logmeanexp(run_res)
        tot_time += t
    end
    return res / n_iters, tot_time/ n_iters
end

function run_experiments()
    n_iters = 20
    dys = [0.7, 1.5, -0.7, 0.35, -1.5, 0.65, 0.7, -0.65, -0.65, 0.3]    
    ys = [sum(dys[1:i]) for i=1:10]
    constraints = choicemap([(:y => i, y) for (i, y) in enumerate(ys)]...)
    tr, _ = generate(model, (10,), constraints)
    obs = [tr[:noisy_pc => t] for t=1:10]
    idx = 1
    times, logzs = [[], []], [[], []]
    for (algo, num_particles) in experiments
        avg_logz, avg_t = avg_weight_experiment(n_iters, num_particles, algo, obs)
        println("$(name[algo]) $num_particles $avg_logz $avg_t")
        println("-----")
        push!(times[idx], avg_t)
        push!(logzs[idx], avg_logz)
        idx = idx == 1 ? 2 : 1
    end
    println("times = " * string(times))
    println("results = " * string(logzs))
end

run_experiments()
