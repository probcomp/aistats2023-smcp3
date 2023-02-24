using Pkg; Pkg.activate(".")

include("boilerplate.jl")
include("noisy_renderer.jl")
include("inference.jl")
include("model.jl")

mean(v) = sum(v) / length(v)
name = Dict(grid_smcp3 => "smcp3", bootstrap_particle_filter => "bspf")
runs = Dict("smcp3" => [], "bspf" => [])

function avg_weight_experiment(n_iters, num_particles, algo, obs)
    res = 0
    push!(runs[name[algo]], [])
    for i=1:n_iters
        run_res = algo(obs, num_particles)[2][end]
        println("run $i of $(name[algo]) with $num_particles particles:" *
                string(run_res))
        push!(runs[name[algo]][end], run_res)
        res += logsumexp(run_res) - log(num_particles)
    end
    return res / n_iters
end

function run_experiments()
    n_iters = 10
    tr = simulate(model, (10,))
    obs = [tr[:noisy_pc => t] for t=1:10]
    for num_particles=2:2:10, algo in keys(name)
        avg_logz = avg_weight_experiment(n_iters, num_particles, algo, obs)
        println("$(name[algo]) $num_particles $avg_logz")
        println("-----")
    end
end

run_experiments()
