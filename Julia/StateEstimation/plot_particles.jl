using CairoMakie
using CairoMakie.Colors

include("ula_hmm.jl")

function style_ax!(ax)
    ax.xgridvisible=false
    ax.ygridvisible=false
    ax.yticksvisible = false; ax.yticklabelsvisible=false
    ax.ylabel="PDF"
    ax.xlabel="x value"
    ax.xticks = [0, 2.5, 5.]
end

function make_plot(y, n_particles)
    f = Figure(resolution=(1500, 600), title="Weighted Particles Proposed at t=1", fontsize=30)
    prior_ax = Axis(f[1, 1], title="Boostrap Particle Filter")
    rm_ax = Axis(f[1, 2], title="Resample-Move SMC")
    smcp3_ax = Axis(f[1, 3], title="SMCP3 SMC")

    map(style_ax!, [prior_ax, rm_ax, smcp3_ax])

    plot_posterior!(prior_ax, y)
    plot_posterior!(rm_ax, y)
    plot_posterior!(smcp3_ax, y)

    plot_prior!(prior_ax, y)
    plot_prior!(rm_ax, y)
    plot_prior!(smcp3_ax, y)

    prior_particles = [prior_generate_and_weight_initial(y) for _=1:n_particles]
    rm_particles = [resamplemove_generate_and_weight_initial(y) for _=1:n_particles]
    smc_particles = [smcp3_generate_and_weight_initial(y) for _=1:n_particles]

    plot_weighted_particles!(prior_ax, y, prior_particles)
    plot_weighted_particles!(rm_ax, y, rm_particles)
    plot_weighted_particles!(smcp3_ax, y, smc_particles)

    # for ax in [prior_ax, rm_ax, smcp3_ax]
    #     lines!(ax, [y, y], [-100, 100], color=RGBA(0, 0, 0.3, 0.2), linewidth=4)
    #     ylims!(ax, (-0.05, .7))
    # end

    return f
end
function plot_weighted_particles!(ax, y, particles)
    xs = map(first, particles)
    logweights = map(last, particles)
    probs = exp.(logweights .- logsumexp(logweights))
    normalizeds = sqrt.(probs ./ maximum(probs))
    for (x, prob) in zip(xs, normalizeds)
        plot_particle!(ax, y, x, prob)
    end
end
function plot_particle!(ax, y, x, prob)
    scatter!(ax, [x], [normalized_pdf(x, y)], markersize=max(10, 70*prob), color=RGBA(0, 0, 0, max(0.2, prob)))
end

function P_y(y)
    std = sqrt(X_STD()^2 + Y_STD()^2)
    return exp(logpdf(normal, y, 0, std))
end
function normalized_pdf(x, y)
    joint_p = exp(Gen.assess(initial_model, (), choicemap((:xₜ, x), (:yₜ, y)))[1])
    return joint_p/P_y(y)
end
function plot_posterior!(ax, y)
    x0 = min(-2, y - 2)
    x1 = max(2, y + 2)
    xs = x0:0.01:x1
    lines!(ax, xs, [normalized_pdf(x, y) for x in xs], linewidth=4, color=RGBA(0, 0, 0, 0.2))
end
function plot_prior!(ax, y)
    x0 = min(-2 , y - 2)
    x1 = max(2, y + 2)
    xs = x0:0.01:x1
    lines!(ax, xs, [1/3 * exp(logpdf(normal, x, 0, X_STD())) for x in xs], linewidth=4, color=RGBA(0.5, 0, 0, 0.1))
end

f = make_plot(5.0, 200)