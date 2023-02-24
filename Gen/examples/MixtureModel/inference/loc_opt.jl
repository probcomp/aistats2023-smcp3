const get_trace = GenTraceKernelDSL.get_trace

@kernel function loc_opt_incorporate(tr, new_records)
    i = length(get_args(tr)[1]) + 1
    Πₜ₋₁ = tr[:partition]
    sorted = sort_by_minidx(Πₜ₋₁)
    merge_traces_and_scores = [
        update(get_trace(tr), (new_records,), (UnknownChange(),), choicemap((:partition, add_to_cluster(sorted, j, i))))[1:2]
        for j=1:length(Πₜ₋₁)
    ]
    (new_trace, new_score) = update(get_trace(tr), (new_records,), (UnknownChange(),), choicemap((:partition, add_singleton(Πₜ₋₁, i))))
    logscores = vcat(map(x -> x[2]::Float64, merge_traces_and_scores), [new_score])
    idx ~ categorical(exp.(logscores .- logsumexp(logscores)))
    if idx > length(merge_traces_and_scores)
        new_tr = new_trace
    else
        new_tr = merge_traces_and_scores[idx][1]
    end

    return new_tr
end

@kernel function k_locopt(tr, new_records)
    loc_opt_tr ~ loc_opt_incorporate(tr, new_records)
    Πₜ = loc_opt_tr[:partition]
    return (choicemap((:partition, Πₜ)), choicemap())
end
@kernel function l_locopt(tr)
    i = length(get_args(tr)[1])
    sorted = sort_by_minidx(tr[:partition])
    idx = find(sorted, length(get_args(tr)[1]))
    Πₜ₋₁ = remove(tr[:partition], i)
    return choicemap((:partition, Πₜ₋₁)), choicemap((:loc_opt_tr => :idx, idx))
end