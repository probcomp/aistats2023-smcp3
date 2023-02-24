using CairoMakie, Colors, ColorSchemes
function colors()
    [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
end
markers() = [:circle for _=1:100]

function indexed_assignment(trace)
    assignment = Array{Int}(undef, length(get_args(trace)[1]))
    for (i, s) in enumerate(trace[:partition])
        for j in s
            assignment[j] = i
        end
    end
    return assignment
end

function visualize_trace(trace; colors=colors())
    f = Figure(resolution=(400,800))
    ax = Axis(f[1, 1], aspect=1/2, xlabel="t", ylabel="x")
    visualize_trace!(ax, trace; colors)
    f
end

function style_ax!(ax, n_datapoints)
    CairoMakie.xlims!(ax, (0, n_datapoints + 1))
    ax.xgridvisible, ax.ygridvisible = false, false
    ax.xticksvisible, ax.xticklabelsvisible = false, false
    # ax.yticksvisible, ax.yticklabelsvisible = false, false
    ax.rightspinevisible = false
    ax.topspinevisible = false
end
function make_first_idx(i, assmt)
    return [
        if i == j
            1
        elseif j < i
            j+1
        elseif j > i
            j
        end
        for j in assmt
    ]
end
function visualize_trace!(ax, trace; colors=colors(), markers=markers())
    assmt = indexed_assignment(trace)
    idx_to_cts = [length([i for i in assmt if i == j]) for j=1:maximum(assmt)]
    max_idx = argmax(idx_to_cts)
    assmt = make_first_idx(max_idx, assmt)

    ordered_colors = [colors[cluster_idx] for cluster_idx in assmt]
    ordered_markers = [markers[cluster_idx] for cluster_idx in assmt]
    data = get_args(trace)[1]
    mn, mx = minimum(data), maximum(data)
    CairoMakie.ylims!(ax, (mn-1, mx + 1))
        ax.yticksvisible, ax.yticklabelsvisible = false, false
    style_ax!(ax, length(data))

    CairoMakie.scatter!(ax,
        1:length(data),
        data;
        color=ordered_colors,
        marker=ordered_markers
    )
end