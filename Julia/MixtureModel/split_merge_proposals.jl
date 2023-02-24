### Split ###

struct SplitAuxRandomness
    chap1::Int
    chap2::Int
    p::Union{Int, Nothing}
    function SplitAuxRandomness(c1, c2, p)
        @assert c1 != c2
        return new(c1, c2, p)
    end
end

assess_split(trace, cluster_to_split_id, i, choices) = 
    propose_split(trace, cluster_to_split_id, i; choices)[2]

"""
    propose_split(tr, c, i; choices=nothing)

Propose a split of the cluster with ID `c` in trace `tr`, given that
`i` is the index of the newest datapoint added to the trace.
If `tr` contains datapoint i, then:
    (1) `i` will be chaparone 1.
    (2) A second datapoint `p` will be chosen to go in the same cluster as datapoint `i`,
    to ensure that we never propose a split with i in a singleton cluster.
    (3) A second chaperone `chap2` will be chosen uniforly from the remaining points.
If `trace` does not contain datapoint `i`, then 2 distinct chaparones will be chosen uniformly at random,
    and we will allow the split move to make either cluster a singleton.

Outputs:
1. Aux randomness.  Either `(chap1, chap2)` if `i` is not in `trace`, or `(p, chap2)` otherwise.
2. The "hypothetical trace" which would be brought into existance after implementing the proposed split.
3. A set containing 2 cluster ids, giving the cluster id for the 2 new post-split clusters.
"""
function propose_split(tr, c, i; choices=nothing)
    tr = copytrace(tr)
    do_assess = !isnothing(choices)
    if do_assess
        aux, newtr, cluster_names = choices
        @assert length(cluster_names) == 2
    end

    log_proposal_pdf = 0.0

    members = collect(tr.clusters[c].members)
    @assert length(members) > 1
    has_new = i in tr.active
    if has_new
        @assert length(members) > 2 "members = $members ; i = $i"
        @assert tr.assignments[i] == c "tr.assignments[i] should be $c but is $(tr.assignments[i])"
    end

    # choose chap1 or p
    if has_new
        chap1 = i
        p_logprobs = normalized_pairing_probs(tr, chap1, members, c)
        p = do_assess ? aux.p : members[rand(Categorical(exp.(p_logprobs)))]
        log_proposal_pdf += p_logprobs[findfirst(members .== p)]
        assigned_indices = [i, p]
    else
        chap1 = do_assess ? aux.chap1 : rand(members)
        log_proposal_pdf -= log(length(members))
        assigned_indices = [chap1]
        p = nothing
    end
    @assert isfinite(log_proposal_pdf)

    # choose chap2
    chap2 = do_assess ? aux.chap2 : rand([j for j in members if !(j in assigned_indices)])
    log_proposal_pdf -= log(length(members) - length(assigned_indices))
    push!(assigned_indices, chap2)

    @assert length(Set(assigned_indices)) == length(assigned_indices) "assigned_indices = $assigned_indices ; members = $members; p = $p; chap1 = $chap1; chap2 = $chap2; i = $i, aux = $(do_assess ? aux : nothing)"
    @assert Set(assigned_indices) ⊆ Set(members) "assigned_indices = $assigned_indices ; members = $members; p = $p; chap1 = $chap1; chap2 = $chap2; i = $i"
    @assert isfinite(log_proposal_pdf)
    @assert chap1 isa Int && chap2 isa Int

    # The chaperones tell us the order of the 2 cluster names.
    if do_assess
        cluster_1_name = newtr.assignments[chap1]
        cluster_2_name = newtr.assignments[chap2]
        @assert cluster_1_name in cluster_names && cluster_2_name in cluster_names "cluster_names = $cluster_names but cluster_1_name, cluster_2_name = $cluster_1_name, $cluster_2_name"
    else
        cluster_1_name, cluster_2_name = gensym(), gensym()
    end

    hypothetical_trace, split_lp = do_split_using_chaperones(tr, chap1, chap2, p, cluster_1_name, cluster_2_name, members, do_assess ? newtr : nothing, can_overwrite_tr=true)
    log_proposal_pdf += split_lp

    return ((
        SplitAuxRandomness(chap1, chap2, p),
        hypothetical_trace,
        Set([cluster_1_name, cluster_2_name])
    ), log_proposal_pdf)
end
function normalized_pairing_probs(tr, chap1, members, c)
    unincorporate_point!(tr, chap1)
    cluster1 = gensym()
    incorporate_point!(tr, chap1, cluster1)
    lps = [
        if mem == chap1
            -Inf
        else
            unincorporate_point!(tr, mem)
            l = conditional_likelihood(tr, cluster1, singleton_cluster(tr, mem))
            incorporate_point!(tr, mem, c)
            l
        end
        for mem in members
    ]
    unincorporate_point!(tr, chap1)
    incorporate_point!(tr, chap1, c)
    return lps .- logsumexp(lps)
end

function do_split_using_chaperones(tr, chap1, chap2, p, cluster_1_name, cluster_2_name, members, newtr; can_overwrite_tr=false)
    log_proposal_pdf = 0.
    if can_overwrite_tr
        hypothetical_trace = tr
    else
        hypothetical_trace = copytrace(tr)
    end
    for member in members
        unincorporate_point!(hypothetical_trace, member)
    end
    incorporate_cluster!(hypothetical_trace, cluster_1_name, singleton_cluster(hypothetical_trace, chap1))
    incorporate_cluster!(hypothetical_trace, cluster_2_name, singleton_cluster(hypothetical_trace, chap2))
    if !isnothing(p)
        incorporate_cluster!(hypothetical_trace, cluster_1_name, singleton_cluster(hypothetical_trace, p))
    end

    # for convenience, I'm not going to trace the score for this shuffle or add the order
    # to the auxiliary randomness; the inverter will exactly cancel out the score so it won't matter
    shuffled = sort(members)
    for member in shuffled
        if !(member in [chap1, chap2, p])
            # Choose whether to incorporate into cluster 1 or cluster 2
            cluster_1_score = crp_log_prior_predictive(hypothetical_trace, cluster_1_name, 1) + conditional_likelihood(hypothetical_trace, cluster_1_name, singleton_cluster(hypothetical_trace, member))
            cluster_2_score = crp_log_prior_predictive(hypothetical_trace, cluster_2_name, 1) + conditional_likelihood(hypothetical_trace, cluster_2_name, singleton_cluster(hypothetical_trace, member))
            log_total       = logsumexp([cluster_1_score, cluster_2_score])
            cluster_1_prob  = exp(cluster_1_score - log_total)
            if (!isnothing(newtr) && newtr.assignments[member] == cluster_1_name) || (isnothing(newtr) && rand() < cluster_1_prob)
                incorporate_point!(hypothetical_trace, member, cluster_1_name)
                log_proposal_pdf += cluster_1_score - log_total
            else
                incorporate_point!(hypothetical_trace, member, cluster_2_name)
                log_proposal_pdf += cluster_2_score - log_total
            end
            @assert isfinite(log_proposal_pdf)
        end
    end
    return (hypothetical_trace, log_proposal_pdf)
end

### Split auxiliary randomness inverter (called `h`) ###
h_assess(i, aux, hypothetical_trace, cluster_ids) =
    h_propose(i, hypothetical_trace, cluster_ids; aux)[2]

function h_propose(i, hypothetical_trace, cluster_ids::Set{Symbol}; aux=nothing)
    @assert length(cluster_ids) == 2
    do_assess = !isnothing(aux)
    log_pdf = 0.

    has_new = i in hypothetical_trace.active
    if has_new
        # propose chap2 is a uniform point from the second cluster
        chap1 = i
        cluster_1_name = hypothetical_trace.assignments[chap1]
        cluster_2_name = only(id for id in cluster_ids if id != cluster_1_name)
        cluster1 = hypothetical_trace.clusters[cluster_1_name]
        cluster2 = hypothetical_trace.clusters[cluster_2_name]

        p = do_assess ? aux.p : rand([j for j in cluster1 if p != chap1])
        chap2 = do_assess ? aux.chap2 : rand([j for j in cluster2])
        log_pdf += -log(length(cluster1) - 1)
        log_pdf += -log(length(cluster2))

        return (SplitAuxRandomness(chap1, chap2, p), log_pdf)
    else
        # Choose an order for the 2 cluster ids (ie. decide which cluster id
        # gets chaparone 1)
        if !do_assess
            ids = collect(cluster_ids)
            if rand() < 0.5
                cluster_1_name, cluster_2_name = ids
            else
                cluster_2_name, cluster_1_name = ids
            end
        else
            chap1, chap2 = aux.chap1, aux.chap2
            cluster_1_name = hypothetical_trace.assignments[chap1]
            cluster_2_name = hypothetical_trace.assignments[chap2]
        end
        log_pdf -= log(2) # for the 2 possible ways we could order the cluster ids
        @assert cluster_1_name in cluster_ids && cluster_2_name in cluster_ids

        # propose chap1 is uniform from the first cluster and chap2 is uniform from the second
        cluster1 = hypothetical_trace.clusters[cluster_1_name]
        cluster2 = hypothetical_trace.clusters[cluster_2_name]
        chap1 = do_assess ? aux.chap1 : rand([j for j in cluster1.members])
        chap2 = do_assess ? aux.chap2 : rand([j for j in cluster2.members])
        log_pdf += -log(length(cluster1)) + -log(length(cluster2))

        return (SplitAuxRandomness(chap1, chap2, nothing), log_pdf)
    end
end

### Merge ###
function enumerate_merges(trace, source_cluster_id)
    merge_scores = Dict()
    cluster_i = trace.clusters[source_cluster_id]
    for j in keys(trace.clusters)
        if j == source_cluster_id
            merge_scores[j] = -Inf # never try to merge with yourself.
        else
            merge_scores[j] = 0.0
            # Compute delta to CRP prior: 
            N = length(trace.active)
            
            cluster_j = trace.clusters[j]
            total = N - length(cluster_j.members)
            at_i = length(cluster_i.members)
            merge_scores[j] -= (log(trace.alpha) - log(total + trace.alpha))
            for k in 1:length(cluster_j.members)
                merge_scores[j] += log(at_i) - log(total + trace.alpha)
                if k > 1
                    merge_scores[j] -= (log(k-1) - log(total + trace.alpha))
                end
                total += 1
                at_i  += 1 
            end
            # Compute delta to likelihood
            merge_scores[j] -= conditional_likelihood(trace, gensym(), cluster_j)
            merge_scores[j] += conditional_likelihood(trace, source_cluster_id, cluster_j)
        end
    end
    return merge_scores
end