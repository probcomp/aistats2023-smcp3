#=
This is a generative function rather than a kernel, to exploit the fact that in
the current Trace Kernel DSL implementation, kernels that sample from Gen Fns
get a trace from the GF, rather than just the return value.  Since we need to 
compute importance weights for calls to this GF, we want the kernel DSL function
that calls `propose_split` to get a trace.
When we unify GFs and Kernel functions, we should have some (more natural,
Prox-like) way to express when we want to sample a trace from a sub-GF, rather
than a return value.
=#
@gen function propose_split(tr, c, i)
    has_new = i ∈ c
    if has_new
        @assert length(c) > 2
    end

    # choose chap1 or p
    if has_new
        chap1 ~ exactly(i)
        collected = collect(setdiff(c, Set([chap1])))
        p_logprobs = normalized_pairing_probs(tr, chap1, collected)
        p ~ categorical_from_list(collected, exp.(p_logprobs))
    else
        chap1 ~ uniform_from_list(collect(c))
        p ~ exactly(nothing)
    end

    # choose chap2
    chap2 ~ uniform_from_list(collect(setdiff(c, Set([chap1, p]))))

    final_partition ~ do_split_using_chaperones(tr, c, chap1, chap2, p)

    return final_partition
end
function normalized_pairing_probs(tr, chap1, other_indices)
    data = get_args(tr)[1]
    @assert length(data) ≥ chap1 "data = $data ; chap1 = $chap1"
    @assert length(data) ≥ maximum(other_indices)
    logscores = [
        get_score(generate(
            get_gen_fn(tr),
            ([data[chap1], data[i]],),
            choicemap((:partition, Set([Set([1, 2])]))),
        )[1])
        for i in other_indices
    ]
    return logscores .- logsumexp(logscores)
end
@gen function do_split_using_chaperones(tr, c, chap1, chap2, p)
    c1 = isnothing(p) ? Set([chap1]) : Set([chap1, p])
    c2 = Set([chap2])
    for i in c
        (i in [chap1, chap2, p]) && continue
        c1_score = get_score(generate(get_gen_fn(tr), get_args(tr), choicemap((:partition, Set([union(c1, Set([i])), c2]))))[1])
        c2_score = get_score(generate(get_gen_fn(tr), get_args(tr), choicemap((:partition, Set([union(c2, Set([i])), c1]))))[1])
        logtotal = logsumexp([c1_score, c2_score])
        is_with_chap1 = {(:is_with_chap1, i)} ~ bernoulli(exp.(c1_score  - logtotal))
        if is_with_chap1
            push!(c1, i)
        else
            push!(c2, i)
        end
    end
    return (
        # Partition
        union(setdiff(tr[:partition], Set([c])), Set([c1, c2])),
        # 2 new clusters, ordered by which one has `:chap1`
        (c1, c2)
        # TODO: maybe I should have it return this unordered, since that is really the interface here...
    )
end

@kernel function sample_split_iterated(tr¹, c¹, i, n_split_proposals)
    split_proposal_traces = []
    split_weights = []
    for j=1:n_split_proposals
        proposal_tr = {(:proposed_split, j)} ~ propose_split(tr¹, c¹, i)
        push!(split_proposal_traces, proposal_tr)
        Π, split_clusters = get_retval(proposal_tr)
        weight = (
            get_score(update(tr¹, choicemap((:partition, Π)))[1]) # Score of trace with split implemented
            + GenTraceKernelDSL.assess(h, (Π, Set(collect(split_clusters)), i), get_h_choices(proposal_tr))[2]  # "meta-inference" score
            - get_score(proposal_tr)                        # proposal score
        )
        @assert !(isinf(weight) || isnan(weight)) """
        get_score(update(tr¹, choicemap((:partition, Π)))[1]) = $(get_score(update(tr¹, choicemap((:partition, Π)))[1]))
        GenTraceKernelDSL.assess(h, (Π, Set(collect(split_clusters)), i), get_h_choices(proposal_tr.trace))[2] = $( GenTraceKernelDSL.assess(h, (Π, Set(collect(split_clusters)), i), get_h_choices(proposal_tr.trace))[2])
        get_score(proposal_tr.trace) = $(get_score(proposal_tr.trace))

        Π = $Π
        Set(collect(split_clusters)), i) = $(Set(collect(split_clusters)))
        get_h_choices(proposal_tr.trace) = $(get_h_choices(proposal_tr.trace))
        """
        push!(split_weights, weight)
    end
    mean_weight = logsumexp(split_weights) .- log(n_split_proposals)
    # println("exp.(split_weights .- logsumexp(split_weights)) = $(exp.(split_weights .- logsumexp(split_weights)))")
    split_idx ~ categorical(exp.(split_weights .- logsumexp(split_weights)))
    (Π_split, two_split_clusters) = get_retval(split_proposal_traces[split_idx])
    return (mean_weight, Π_split, two_split_clusters)
end

### Split auxiliary randomness inverter (called `h`) ###
@kernel function h(Π_after_split, unordered_clusters, i)
    @assert length(Π_after_split) ≥ 2 "Π_after_split = $Π_after_split"
    has_new = i in union(Π_after_split...)
    if has_new
        chap1 ~ exactly(i)
        c1 = cluster_containing(unordered_clusters, chap1)
        c2 = only(setdiff(unordered_clusters, Set([c1])))
        p ~ uniform_from_list(collect(setdiff(c1, Set([chap1]))))
        chap2 ~ uniform_from_list(collect(c2))
    else
        chap1 ~ uniform_from_list(collect(union(unordered_clusters...)))
        c2 = only(setdiff(unordered_clusters, Set([cluster_containing(unordered_clusters, chap1)])))
        chap2 ~ uniform_from_list(collect(c2))
        p ~ exactly(nothing)
    end

    return (chap1, chap2, p)
end

function get_h_choices(propose_split_tr)
    choicemap(
        (:p, propose_split_tr[:p]),
        (:chap1, propose_split_tr[:chap1]),
        (:chap2, propose_split_tr[:chap2])
    )
end

### Functions for filling in splitting randomness in reverse moves. ###

@kernel function sample_randomness_for_reverse_split(pre_move_trace, post_move_Π, premove_cluster_to_split, i, n_split_proposals, fwd_move)
    cm = choicemap()
    if fwd_move <= 2
        # If the first direction did a stay or split, the reverse did not do a split,
        # so we can just sample a split from exactly the same kernel.
        split ~ TracedKernel(sample_split_iterated)(pre_move_trace, premove_cluster_to_split, i, n_split_proposals)
        (_, choices, _) = split
        return choices
    else
        # If the first direction did a merge, the reverse did a split.  So we need to 
        # constrain the selected split to be consistent with the produced trace.
        splits = Any[nothing for _=1:n_split_proposals]
        split_idx ~ uniform_discrete(1, n_split_proposals)
        cm[:split_idx] = split_idx
        for j=1:n_split_proposals
            j == split_idx && continue
            splits[j] = {(:splits, i)} ~ propose_split(pre_move_trace, premove_cluster_to_split, i)
            set_submap!(cm, (:proposed_split, j), get_choices(splits[j]))
        end

        split_clusters = post_split_clusters(post_move_Π, union(premove_cluster_to_split, Set([i])))
        if length(split_clusters) != 2
            println("post_move_Π: $(post_move_Π)")
            println("premove_cluster_to_split: $premove_cluster_to_split")
            println()
            error()
        end
        aux ~ h(post_move_Π, split_clusters, i)
        (chap1, chap2, p) = aux
        set_submap!(cm, (:proposed_split, split_idx) => :final_partition, inner_split_decision_submap(post_move_Π, premove_cluster_to_split, chap1, chap2, p))
        cm[(:proposed_split, split_idx) => :chap1] = chap1
        cm[(:proposed_split, split_idx) => :chap2] = chap2
        cm[(:proposed_split, split_idx) => :p] = p
    end
    return cm
end
function inner_split_decision_submap(post_move_Π, premove_cluster_to_split, chap1, chap2, p)
    cm = choicemap()
    c1 = cluster_containing(post_move_Π, chap1)
    for j in premove_cluster_to_split
        j in [chap1, chap2, p] && continue
        cm[(:is_with_chap1, j)] = j in c1
    end
    return cm
end

# Randomness from the `sample_split_iterated` to `sample_randomness_for_reverse_split`.
function splitrand_to_revrand(split_choices, splitter_move_idx, n_split_proposals)
    cm = choicemap()
    if splitter_move_idx != 2
        set_submap!(cm, :split, split_choices)
    else
        cm[:split_idx] = split_choices[:split_idx]
        for j=1:n_split_proposals
            if j != split_choices[:split_idx]
                set_submap!(cm, (:splits, j), get_submap(split_choices, (:proposed_split, split_choices[:split_idx])))
            end
        end
        set_submap!(cm, :aux, get_h_choices(get_submap(split_choices, (:proposed_split, split_choices[:split_idx]))))
        # TODO: fill in the `final_partition` chocies
    end
    return cm
end

### Merge partitions + scores ###
function get_merge_partitions_scores_and_pairs(tr, c)
    partitions, traces, scores, pairs = [], [], [], []
    for s in sort_by_minidx(tr[:partition])
        if s != c
            Π = merge(tr[:partition], s, c)
            trnew, score = update(tr, choicemap((:partition, Π)))[1:2]
        else
            Π, trnew, s = nothing, nothing, nothing
            score = -Inf
        end
        push!(partitions, Π)
        push!(traces, trnew)
        push!(scores, score)
        push!(pairs, (c, s))
    end
    return collect(zip(partitions, scores, traces, pairs))
end