using CairoMakie


results = ([-133731.70430841585, -32679.816155666707, -5716.375418014269, -10667.620389664513, -8690.45078907437],
           [-2686.7635500601823, -982.4300496663855, -367.7916726117488, 171.4645640512923, 1045.7267060339818])

function make_logz_plot(results, nparticles_range)
    w = 600
    h = Int(ceil(1.3w))
    f = Figure(resolution=(w,h), fontsize=38)
    ax = Axis(f[1, 1])
    ax.xlabel = "Number of SMC Particles"
    ax.ylabel = "Average estimate of log P(y₁..ₜ) × 10⁻⁴"
    ax.xgridvisible = false
    ax.ygridvisible = false

    lines!(nparticles_range, results[1] ./ 1e4, label="Particle Filter", color=:red2)
    lines!(nparticles_range, results[2] ./ 1e4, label="SMCP3", color=:black)

    axislegend(ax, position=:rb)

    f
end

save("cairo.svg", make_logz_plot(results, 2:2:10))
