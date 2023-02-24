using CairoMakie, Colors, ColorSchemes
function colors()
    # [:grey30, :blue3, :red3, :green]
    ColorSchemes.tol_bright
end
markers() = [:circle, :rect, :diamond, :hexagon] #:circle for _=1:100]

function indexed_assignment(trace)
    clusterid_to_idx = Dict()
    for (i, cid) in enumerate(keys(trace.clusters))
        clusterid_to_idx[cid] = i
    end

    return [
        clusterid_to_idx[trace.assignments[i]] for i=1:length(trace.assignments) if i in trace.active
    ]
    # vals = []
    # for i=1:length(trace.assignments)
    #     if i in trace.active
    #         push!(vals, clusterid_to_idx[trace.assignments[i]])
    #     end
    # end
    # #     clusterid_to_idx[cid]
    # #     for (i, cid) in enumerate(trace.assignments)
    # #         if i in trace.active
    # # ]
    # return vals
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
function visualize_trace!(ax, trace::DPMMTrace; colors=colors(), markers=markers())
    assmt = indexed_assignment(trace)
    idx_to_cts = [length([i for i in assmt if i == j]) for j=1:maximum(assmt)]
    max_idx = argmax(idx_to_cts)
    assmt = make_first_idx(max_idx, assmt)

    ordered_colors = [colors[cluster_idx] for cluster_idx in assmt]
    ordered_markers = [markers[cluster_idx] for cluster_idx in assmt]
    mn, mx = minimum(trace.data), maximum(trace.data)
    CairoMakie.ylims!(ax, (mn-1, mx + 1))
        ax.yticksvisible, ax.yticklabelsvisible = false, false
    style_ax!(ax, length(trace.data))

    CairoMakie.scatter!(ax,
        1:length(trace.active),
        trace.data[1:length(trace.active)];
        color=ordered_colors,
        marker=ordered_markers
    )
end