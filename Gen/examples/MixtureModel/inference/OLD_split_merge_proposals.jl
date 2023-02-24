@kernel function propose_split(tr, c, i)
    has_new = i ∈ c
    if has_new
        @assert length(c) > 2
    end

    # choose chap1 or p
    if has_new
        chap1 ~ exactly(i)
        collected = collect(setdiff(c, chap1))
        p_logprobs = normalized_pairing_probs(tr, chap1, collected)
        p ~ categorical_from_list(collected, exp.(p_logprobs))
    else
        chap1 ~ uniform_from_list(collect(setdiff(c, Set([i]))))
        p ~ exactly(nothing)
    end

    # choose chap2
    chap2 ~ uniform_from_list(collect(setdiff(c, Set([chap1, chap2]))))

    Π2_trace ~ do_split_using_chaperones(tr, c, chap1, chap2, p)

    return get_retval(Π2_trace)
end
function normalized_pairing_probs(tr, chap1, other_indices)
    data = get_args(tr)[1]
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

@kernel function do_split_using_chaperones(tr, c, chap1, chap2, p)
    c1 = isnothing(p) ? Set([chap1]) : Set([chap1, p])
    c2 = Set([chap2])
    for i in c
        (i in [chap1, chap2, p]) && continue
        c1_score = nothing
        c2_score = nothing
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
        # 2 new clusters
        (c1, c2)
    )
end

@kernel function sample_split_iterated(tr¹, c¹, i, n_split_proposals)
    split_proposal_traces = []
    split_weights = []
    for j=1:n_split_proposals
        proposal_tr = {(:proposed_split, j)} ~ propose_split(tr¹, c¹, i)
        push!(split_proposal_traces, proposal_tr)

        weight = (
            assess(h, (), get_h_choices(proposal_tr))[3] -
            assess(propose_split, (tr¹, c¹, i), get_choices(proposal_tr.trace))[2]
        )
        push!(k_split_weights, weight)
    end
    mean_weight = logsumexp(split_weights) .- log(n_split_proposals)
    split_idx ~ categorical(exp.(split_weights .- logsumexp(split_weights)))
    (Π_split, two_split_clusters) = get_retval(split_proposal_traces[split_idx])
    return (mean_weight, Π_split, two_split_clusters)
end

### Split auxiliary randomness inverter (called `h`) ###
@kernel function h()
    has_new = i in 
    # TODO

    return (chap1, chap2, p)
end

### Merge partitions + scores ###
function get_merge_partitions_and_scores(tr, c)
    partitions, traces, scores = [], [], []
    for s in tr[:partition] if s != c
        Π = merge(tr[:partition], s, c)
        trnew, score = update(tr, choicemap((:partition, Π)))[1:2]
        push!(partitions, Π)
        push!(traces, trnew)
        push!(scores, score)
    end
    return (partitions, scores, traces)
end








#=
for j=1:n_split_proposals
    k_splits[j] = {(:k_splits, i)} ~ propose_split(trₜ₋₁, cₜ₋₁, i)
    set_submap!(cm, :k_split_tr => (:proposed_split, j), get_choices(k_splits[j]))    
end
k_split_idx ~ categorical()


if length(cₜ₋₁) >= 2
    k_splits = Any[nothing for _=1:n_split_proposals]
    if move > 2
        # If the first direction did a stay or split, the reverse did not do a split,
        # so we can just sample a split from exactly the same kernel.
        k_split ~ sample_split_iterated(trₜ₋₁, cₜ₋₁, i)
        set_submap!(cm, k_split_tr, get_choices(k_split))
    else
        # If the first direction did a merge, the reverse did a split.  So we need to 
        # constrain the selected split to be consistent with the produced trace.
        k_split_idx ~ uniform_discrete(1:n_split_proposals)
        cm[:k_split_tr => (:proposed_split, k_split_idx) => :split_idx] = k_split_idx
        for j=1:n_split_proposals
            j == k_split_idx && continue
            k_splits[j] = {(:k_splits, i)} ~ propose_split(trₜ₋₁, cₜ₋₁, i)
            set_submap!(cm, :k_split_tr => (:proposed_split, j), get_choices(k_splits[j]))    
        end
        aux ~ h()
        (chap1, chap2, p) = get_retval(aux)
        cm[:k_split_tr => (:proposed_split, k_split_idx) => :chap1] = chap1
        cm[:k_split_tr => (:proposed_split, k_split_idx) => :chap2] = chap2
        cm[:k_split_tr => (:proposed_split, k_split_idx) => :p] = p
        fill_in_split_decisions!(cm, :k_split_tr => (:proposed_split, k_split_idx), trₜ, cₜ₋₁)
    end
end

=#