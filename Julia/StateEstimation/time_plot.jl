include("gen_ula_hmm.jl")
using CairoMakie, Distributions, Colors

# there are a couple dependencies in here...
include("../MixtureModel/gaussian/_deps_for_timeplot.jl")

function get_fax(n_ax=1)
    f = Figure(resolution=(650, 50 + 200*n_ax), fontsize=18)

    axes = []
    for i=1:n_ax
        ax = Axis(f[i, 1], xlabel="t", ylabel="x")
        ax.xgridvisible = false
        ax.ygridvisible = false
        ax.xticksvisible = false
        ax.yticksvisible = false
        ax.xticklabelsvisible = false
        ax.yticklabelsvisible = false 
        ax.rightspinevisible = false
        ax.topspinevisible = false

        push!(axes, ax)
    end
    return (f, axes)
end

function plot_obs!(ax, ys)
    lines!(ax, 1:length(ys), ys, color=:black, linestyle=:dash)
end

# function get_particle_positions_plot(ys)
#     (f, ax) = get_fax()
#     obs = plot_obs!(ax, ys)
#     N = 100
#     smcp3_endstate = smc(ys, N, initial_proposal=initial_smcp3_ula_proposal, step_proposal=step_smcp3_ula_proposal)
#     particles = Gen.sample_unweighted_traces(smcp3_endstate, 100)
#     pts = nothing
#     for tr in particles
#         println("scattering $([get_xy(tr, t)[1] for t=0:length(ys)-1])")
#         pts = Makie.scatter!(ax, 1:length(ys), [get_xy(tr, t)[1] for t=0:length(ys)-1], color=RGBA(0, 0, 0, 0.1))
#     end

#     smcp3_endstate = smc(ys, N)
#     particles = Gen.sample_unweighted_traces(smcp3_endstate, 100)
#     pts2 = nothing
#     for tr in particles
#         println("scattering $([get_xy(tr, t)[1] for t=0:length(ys)-1])")
#         pts2 = Makie.scatter!(ax, 1:length(ys), [get_xy(tr, t)[1] for t=0:length(ys)-1], color=RGBA(1, 0, 0, 0.1))
#     end

#     axislegend(ax, [obs, MarkerElement(color = RGBA(0, 0, 0, 0.8), marker = :circle)], ["Observed noisy trajectory", "Inferred particles"], position=:rb)
#     f
# end

function kalman(ys)
    mus = []
    vars = []
    mu = 0
    var = 1.
    
    for y in ys
        K = (1 + var)/(1 + 1 + var)
        mu_new = mu + K*(y - mu)
        var_new = K

        push!(mus, mu_new)
        push!(vars, var_new)
        mu, var = mu_new, var_new
    end

    return mus, vars
end


function normalized(weights)
    exp.(weights .- logsumexp(weights))
end

function get_mean_positions_plot(ys, ax)
    N = 5
    style_ax!(ax, length(ys))

    pf_states = [smc(ys, N, get_intermediaries=true)[2] for _ in 1:5]
    rm_states = [smc(ys, N, mcmc_step=my_mala, get_intermediaries=true)[2] for _ in 1:5]
    smcp3_states = [smc(ys, N, initial_proposal=initial_smcp3_ula_proposal, step_proposal=step_smcp3_ula_proposal, get_intermediaries=true)[2] for _ in 1:5]

#    println(typeof(pf_states[1]))

    # Compute mean position at each time step for each algorithm for each run.
    # Note that pf_states is a map from (run_id) to (time) to (particles, weights).
    pf_means = [ [mean(exp(w) * get_xy(tr, t-1)[1] for (w, tr) in zip(normalized(weights), traces))  for (t, (traces, weights)) in enumerate(pf_state)] for pf_state in pf_states]
    rm_means = [ [mean(exp(w) * get_xy(tr, t-1)[1] for (w, tr) in zip(normalized(weights), traces)) for (t, (traces, weights)) in enumerate(rm_state)] for rm_state in rm_states]
    smcp3_means = [ [mean(exp(w) * get_xy(tr, t-1)[1] for (w, tr) in zip(normalized(weights), traces))  for (t, (traces, weights)) in enumerate(smcp3_state)] for smcp3_state in smcp3_states]

    #f = Figure(resolution=(500, 150))
    #ax = Axis(f[1,1], title="Expected position")
    ax.xticksvisible=false
    ax.xticklabelsvisible=false
    pf_mean_means = [mean(pf_means[i][t] for i in 1:5) for t in 1:length(ys)]
    rm_mean_means = [mean(rm_means[i][t] for i in 1:5) for t in 1:length(ys)]
    smcp3_mean_means = [mean(smcp3_means[i][t] for i in 1:5) for t in 1:length(ys)]
    CairoMakie.lines!(ax, 1:length(ys), pf_mean_means; color=:red)
    CairoMakie.lines!(ax, 1:length(ys), rm_mean_means; color=:blue)
    CairoMakie.lines!(ax, 1:length(ys), smcp3_mean_means; color=:black)

    for ms in pf_means
        # Red with low alpha
        CairoMakie.lines!(ax, 1:length(ys), ms; color=RGBA(1, 0, 0, 0.5))
    end
    for ms in rm_means
        # Blue with low alpha
        CairoMakie.lines!(ax, 1:length(ys), ms; color=RGBA(0, 0, 1, 0.5))
    end
    for ms in smcp3_means
        # with lower alpha:
        CairoMakie.lines!(ax, 1:length(ys), ms; color=RGBA(0, 0, 0, 0.6))
    end
    true_means = [Float64(mu) for mu in kalman(ys)[1]]
   # CairoMakie.lines!(ax, 1:length(ys), true_means; color=:green)

    # println([mean(rm_means[i][t] for i in 1:5) for t in 1:length(ys)])
    # println([mean(smcp3_means[i][t] for i in 1:5) for t in 1:length(ys)])
    return ax
end

# ys = [i + normal(0, .8) for i in
#     [0, 3, 6, 9, 12, 9, 12, 16, 18, 18, 20, 17]]
# f = get_mean_positions_plot(ys .* 2)
# save("tracked_mean_ula.pdf", f)


function get_logz_plot(ys, ax)

    N = 5
    
    _, smcp3_states = smc(ys, N, get_intermediaries=true)

    gibbs_lml_ests = [logsumexp(weights) for (_, weights) in smcp3_states]

    _, smcp3_states = smc(ys, N, mcmc_step=my_mala, get_intermediaries=true)

    rm_lml_ests = [logsumexp(weights) for (_, weights) in smcp3_states]

    _, smcp3_states = smc(ys, N, initial_proposal=initial_smcp3_ula_proposal, step_proposal=step_smcp3_ula_proposal, get_intermediaries=true)

    smcp3_lml_ests = [logsumexp(weights) for (_, weights) in smcp3_states]
    style_ax!(ax, 12)


    # ax2.ygridvisible = true
    ax.xticksvisible = true
    ax.yticksvisible = true
    ax.xticklabelsvisible=true
    ax.yticklabelsvisible = true
    redline = lines!(ax, 1:length(ys), gibbs_lml_ests, color=:red)
    blueline = lines!(ax, 1:length(ys), rm_lml_ests, color=:blue)
    blackline = lines!(ax, 1:length(ys), smcp3_lml_ests, color=:black)
    # axislegend(ax2, [
    #     MarkerElement(color = RGBA(0, 0, 0, 0.8), marker = :circle),
    #     MarkerElement(color = RGBA(1, 0, 0, 0.8), marker = :circle)
    # ], ["SMCP3", "Bootstrap PF"], position=:rt)

    # l = Legend(f[3, 1], [obs, bnd, MarkerElement(color = RGBA(0, 0, 0, 0.8), marker = :circle, markersize=M),  MarkerElement(color = RGBA(1, 0, 0, 0.8), marker = :circle, markersize=M), MarkerElement(color = RGBA(0, 0, 1, 0.8), marker = :circle, markersize=M)],
    #     ["Noisy\nobserved\ntrajectory", "1σ window\n of exact\nfiltering\nposterior", "SMCP3", "Boostrap\nPF", "Resample-\nMove SMC"],
    #     labelsize=16,
    #     orientation=:horizontal
    # )
    # l.tellwidth= false
    # l.tellheight = true

    return (ax, redline, blueline, blackline)

end

# f = get_logz_plot(ys)
# save("logz_ula.pdf", f)


function get_particle_positions_plot(ax)
   # f = Figure(resolution=(500, 150)) #    (f, (ax, ax2)) = get_fax(2)
   # ax = Axis(f[1, 1])
    obs = plot_obs!(ax, ys)
    N = 5
    M = 12
    MKR = :circle
    BLURVAR = 0.

    # _, smcp3_states = smc(ys, 100000, initial_proposal=initial_smcp3_ula_proposal, step_proposal=step_smcp3_ula_proposal, get_intermediaries=true)

    _, smcp3_states = smc(ys, N, get_intermediaries=true)
    # for t in keys(smcp3_states)
    #     (particles, log_weights) = smcp3_states[t]
    #     resampled_particles = [
    #         particles[rand(Categorical(exp.(log_weights .- logsumexp(log_weights))))]
    #         for _=1:100
    #     ]
        
    #     # println("scattering $([get_xy(tr, t)[1] for t=0:length(ys)-1])")
    #     # pts = Makie.scatter!(ax, 1:length(ys), [get_xy(tr, t)[1] for t=0:length(ys)-1], color=RGBA(0, 0, 0, 0.1))
    #     pts = Makie.scatter!(ax, [t + normal(0, BLURVAR) for _=1:100], [get_xy(tr, t-1)[1] for tr in resampled_particles[1:1]], color=RGBA(1, 0, 0, 0.1), markersize=M, marker=MKR)
    # end
    gibbs_lml_ests = [logsumexp(weights) for (_, weights) in smcp3_states]

    _, smcp3_states = smc(ys, N, mcmc_step=my_mala, get_intermediaries=true)
    # for t in keys(smcp3_states)
    #     (particles, log_weights) = smcp3_states[t]
    #     resampled_particles = [
    #         particles[rand(Categorical(exp.(log_weights .- logsumexp(log_weights))))]
    #         for _=1:100
    #     ]
        
    #     # println("scattering $([get_xy(tr, t)[1] for t=0:length(ys)-1])")
    #     # pts = Makie.scatter!(ax, 1:length(ys), [get_xy(tr, t)[1] for t=0:length(ys)-1], color=RGBA(0, 0, 0, 0.1))
    #     pts = Makie.scatter!(ax, [t + normal(0, BLURVAR) for _=1:100], [get_xy(tr, t-1)[1] for tr in resampled_particles], color=RGBA(0, 0, 1, 0.1), markersize=M, marker=MKR)
    # end
    rm_lml_ests = [logsumexp(weights) for (_, weights) in smcp3_states]

    _, smcp3_states = smc(ys, N, initial_proposal=initial_smcp3_ula_proposal, step_proposal=step_smcp3_ula_proposal, get_intermediaries=true)
    # particles = Gen.sample_unweighted_traces(, 100)
    pts = nothing
    for t in keys(smcp3_states)
        (particles, log_weights) = smcp3_states[t]
        resampled_particles = [
            particles[rand(Categorical(exp.(log_weights .- logsumexp(log_weights))))]
            for _=1:100
        ]
        
        # println("scattering $([get_xy(tr, t)[1] for t=0:length(ys)-1])")
        # pts = Makie.scatter!(ax, 1:length(ys), [get_xy(tr, t)[1] for t=0:length(ys)-1], color=RGBA(0, 0, 0, 0.1))
        pts = Makie.scatter!(ax, [t + normal(0, BLURVAR) for _=1:100], [get_xy(tr, t-1)[1] for tr in resampled_particles[1:1]], color=RGBA(0, 0, 0, 0.1), markersize=M, marker=MKR)
    end
    smcp3_lml_ests = [logsumexp(weights) for (_, weights) in smcp3_states]
    #ax.yticksvisible, ax.yticklabelsvisible = false, false
    style_ax!(ax, 12)

    # CairoMakie.scatter!(ax,
    #     1:length(trace.active),
    #     trace.data[1:length(trace.active)];
    #     color=ordered_colors,
    #     marker=ordered_markers
    # )

    means, vars = kalman(ys)
    topline = means .+ 3 * sqrt.(vars)
    botline = means .- 3 * sqrt.(vars)
    # lines!(ax, 1:length(ys), topline, color=:orange)
    # lines!(ax, 1:length(ys), botline, color=:orange)
    bnd = band!(ax, 1:length(ys), botline, topline; color = (:black, 0.2))

    # smcp3_endstate = smc(ys, N)
    # particles = Gen.sample_unweighted_traces(smcp3_endstate, 100)
    # pts2 = nothing
    # for tr in particles
    #     println("scattering $([get_xy(tr, t)[1] for t=0:length(ys)-1])")
    #     pts2 = Makie.scatter!(ax, 1:length(ys), [get_xy(tr, t)[1] for t=0:length(ys)-1], color=RGBA(1, 0, 0, 0.1))
    # end

    # axislegend(ax,
    #     [obs, [
    #         MarkerElement(color = RGBA(0, 0, 0, 0.8), marker = :circle),
    #         MarkerElement(color = RGBA(1, 0, 0, 0.8), marker = :circle)
    #     ]],
    #     ["Observed noisy trajectory", "Inferred particles"], position=:rb
    # )

    # ax2.ylabel = "Est. of Log P(y₁..ₜ)"
    # # ax2.ygridvisible = true
    # ax2.yticksvisible = true
    # ax2.yticklabelsvisible = true
    # lines!(ax2, 1:length(ys), gibbs_lml_ests, color=:red)
    # lines!(ax2, 1:length(ys), rm_lml_ests, color=:blue)
    # lines!(ax2, 1:length(ys), smcp3_lml_ests, color=:black)
    # # axislegend(ax2, [
    # #     MarkerElement(color = RGBA(0, 0, 0, 0.8), marker = :circle),
    # #     MarkerElement(color = RGBA(1, 0, 0, 0.8), marker = :circle)
    # # ], ["SMCP3", "Bootstrap PF"], position=:rt)

    # l = Legend(f[3, 1], [obs, bnd, MarkerElement(color = RGBA(0, 0, 0, 0.8), marker = :circle, markersize=M),  MarkerElement(color = RGBA(1, 0, 0, 0.8), marker = :circle, markersize=M), MarkerElement(color = RGBA(0, 0, 1, 0.8), marker = :circle, markersize=M)],
    #     ["Noisy\nobserved\ntrajectory", "1σ window\n of exact\nfiltering\nposterior", "SMCP3", "Boostrap\nPF", "Resample-\nMove SMC"],
    #     labelsize=16,
    #     orientation=:horizontal
    # )
    # l.tellwidth= false
    # l.tellheight = true

    #ax.title = "Particles from Inference"
    #ax2.title = "Log Marginal Likelihood Estimates from Inference"
    ax.xlabelvisible=true
    # colsize!(f.layout, 2, Relative(0.2))

    ax, bnd, obs
end

ys = [i + normal(0, .8) for i in
    [0, 3, 6, 9, 12, 9, 12, 16, 18, 18, 20, 17]]
# f = get_particle_positions_plot(ys)

# save("smcp3particle_ula.pdf", f)




function combined_plot(ys)

    f = Figure(resolution=(500, 500))
    M = 12
    MKR = :circle
    viz_ax = Axis(f[1, 1], ylabel="position zₜ")
    mean_ax = Axis(f[2, 1], ylabel="E[zₜ]")
    lml_ax = Axis(f[3, 1], ylabel="log p(y₁..ₜ)", xlabel="time t")
    _, bnd, obs = get_particle_positions_plot(viz_ax)
    get_mean_positions_plot(ys, mean_ax)
    _, redline, blueline, blackline = get_logz_plot(ys, lml_ax)
    l = Legend(f[4, 1], [obs, 
        bnd, 
        MarkerElement(color = RGBA(0, 0, 0, 0.8), marker = :circle, markersize=M), 
        blackline, #MarkerElement(color = RGBA(0, 0, 0, 0.8), marker = :circle, markersize=M),  
        redline, #MarkerElement(color = RGBA(1, 0, 0, 0.8), marker = :circle, markersize=M), 
        blueline], #MarkerElement(color = RGBA(0, 0, 1, 0.8), marker = :circle, markersize=M)],
        ["Noisy observed\ntrajectory", 
         "3σ window of\nexact posterior", 
         "Latent inferred\nby SMCP3",
         "SMCP3", 
         "Boostrap\nfilter", 
         "Resample-\nMove SMC"],
        labelsize=16,
        orientation=:horizontal,
        nbanks=2
    )
    # l.tellwidth= false
    # l.tellheight = true
    f

end

f = combined_plot(ys)

save("combined_ula.pdf", f)




