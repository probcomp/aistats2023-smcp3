using CairoMakie

times = [[4.67828374005, 10.761620287499998, 16.0827726176, 21.3366931121,
          26.5518150176],
         [2.7557157646, 4.6236276506500005, 13.466894990749998,
          16.810814010500003, 21.60445849115, 29.425506773450003]]
results = [[-1008.8811385590252, 2097.1594101813084, 2417.483417005082,
            2695.4376832904245, 2715.690609122592],
           [-3344.531276276234, -835.965090436983, 928.4878724339702,
            1341.8960151127926,1443.6433050673422, 1816.8928734886886]]

function make_logz_plot(results, nparticles_range)
    w = 600
    h = Int(ceil(1.3w))
    f = Figure(resolution=(w,h), fontsize=38)
    ax = Axis(f[1, 1])
    ax.xlabel = "Time (s)"
    ax.ylabel = "Average estimate of log P(y₁..ₜ) × 10⁻³"
    ax.xgridvisible = false
    ax.ygridvisible = false

    lines!(times[2], results[2] ./ 1e3, label="Particle Filter", color=:red2, linewidth=8)
    lines!(times[1], results[1] ./ 1e3, label="SMCP3", color=:black, linewidth=8)

    axislegend(ax, position=:rb)

    f
end

save("cairo.svg", make_logz_plot(results, 2:2:10))
