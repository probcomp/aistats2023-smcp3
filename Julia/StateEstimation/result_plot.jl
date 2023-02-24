#=
Results plot.
X axis = number of particles.
Y axis = average log(z) estimate.

Can later implement exact inference and then compare error in logZ estimate.
=#

using CairoMakie
include("gen_ula_hmm.jl")

function results_for_ys(ys, nparticles_range, n_iterates)
    results = [
        average_logz_experiments(ys, n_particles, n_iterates)
        for n_particles in nparticles_range
    ]
    return ([map(x -> x[i], results) for i=1:3])
end
function results_for_multiple_ys(yss, nparticles_range, n_iterates)
    resultss = [results_for_ys(ys, nparticles_range, n_iterates) for ys in yss]
    return [mean([results[i] for results in resultss]) for i=1:3]
end
results_for_random_ys_from_model(n_runs, T, nparticles_range, n_iterates) =
    results_for_multiple_ys(
        [get_ys(generate(model, (T,))[1]) for _=1:n_runs], nparticles_range, n_iterates
    )

function make_logz_plot(results, nparticles_range)
    f = Figure(resolution=(400,400), fontsize=20)
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
make_logz_plot_for_single_y_sequence(ys, nparticles_range, n_iterates) =
    make_logz_plot(results_for_ys(ys, nparticles_range, n_iterates), nparticles_range)

make_logzplot_for_multiple_y_sequences(yss, nparticles_range, n_iterates) =
    make_logz_plot(results_for_multiple_ys(yss, nparticles_range, n_iterates), nparticles_range)

make_logzplot_for_random_ys_from_model(n_runs, T, nparticles_range, n_iterates) =
    make_logz_plot(results_for_random_ys_from_model(n_runs, T, nparticles_range, n_iterates), nparticles_range)

# f = make_logzplot_for_multiple_y_sequences([[1, 3, 5], [1, 2, 3]], 1:5, 200)
# @time (f = make_logzplot_for_random_ys_from_model(20, 10, 1:20, 200))

@time (results = results_for_random_ys_from_model(20, 10, 1:20, 200))
f = make_logz_plot(results, 1:20)
