using Random: shuffle

include("locally-optimal-smc.jl")
include("split_merge_proposals.jl")

### Top-level SMC algorithm ###

struct SMCP3Options
    ess_threshold :: Float64
    n_split_proposals :: Int
end

function run_smcp3(data::Vector, K::Int, hypers::Hyperparameters, C::Type, alpha, options=SMCP3Options(K * 1/5, 10); retained=nothing, return_record=false)
    @assert isnothing(retained) "SMCP3 kernel does not currently support CSMC"

    traces = [create_initial_dpmm_trace(hypers, C, data, alpha) for _ in 1:K]
    weights = zeros(K)

    if return_record
        trace_record, weight_record = [], []
    else
        trace_record, weight_record = nothing, nothing
    end

    for i in 1:length(data)
        # Threads.@threads
        for j = 1:length(traces)
            trace = traces[j]
            newtrace, deltaweight = smcp3_step!(trace, i; n_split_proposals=options.n_split_proposals)
            traces[j] = newtrace
            weights[j] += deltaweight
        end

        if return_record
            push!(trace_record, copy(traces))
            push!(weight_record, copy(weights))
        end

        (traces, weights) = maybe_resample(traces, weights, options.ess_threshold; retained)
    end

    return ((traces, weights), (trace_record, weight_record))
end

function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    return exp(log_ess)
end

function maybe_resample(traces, weights, ess_threshold; retained=nothing)
    K = length(traces)
    K_unretained = isnothing(retained) ? K : K-1

    total_weight = logsumexp(weights)
    log_normalized_weights = weights .- total_weight
    ess = effective_sample_size(log_normalized_weights)
    if ess < ess_threshold
        normalized_weights = exp.(log_normalized_weights)
        indices = [rand(Categorical(normalized_weights)) for _ in 1:K_unretained]
        !isnothing(retained) && push!(indices, K)
        traces = [copytrace(traces[i]) for i in indices]
        weights = fill(total_weight - log(K), K)
    end
    return (traces, weights)
end

### SMCP3 Kernel ###

function smcp3_step!(trace, i; n_split_proposals=1, retained=nothing)
    @assert isnothing(retained) "CSMC not supported for SMCP3 step."
    (newtrace, values_for_L, log_k_pdf) = k_propose(trace, i, n_split_proposals)
    log_l_pdf = l_assess(newtrace, i, values_for_L, n_split_proposals)
    log_weight_update = log_joint(newtrace) - log_joint(trace) + log_l_pdf - log_k_pdf

    # println("""
    # log_joint(newtrace) = $(log_joint(newtrace))
    # log_joint(trace) = $(log_joint(trace))
    # log_l_pdf = $log_l_pdf
    # log_k_pdf = $log_k_pdf
    # """)

    return (newtrace, log_weight_update)
end

#=
K consumes/produces the following traces:
1. tr????????? : Previous trace, without the new datapoint.
2. tr?????  : (1) + the new datapoint is incorporated into a cluster.
3. tr?????  : (2) + the cluster containing the new datapoint has been split.
4. tr?????  : (1) + either a split move, merge move, or stay move has been performed.

Datapoint indices:
1. i : index of new datapoint to incorporate

Cluster IDs:
1. c????? : ID of the cluster `i` was incorporated into in tr?????
=#
function k_propose(tr?????????, i, n_split_proposals)
    log_k_pdf = 0.0

    tr????? = copytrace(tr?????????)
    log_k_pdf += locally_optimal_smc_step!(tr?????, i; get_proposal_probability = true)
    
    c????? = tr?????.assignments[i]
    c?????_members = tr?????.clusters[c?????].members
    if length(c?????_members) == 1
        # println("returning after smc step.")
        # println("after smc step, log_k_pdf is $log_k_pdf")
        return (tr?????, nothing, log_k_pdf)
    end

    if length(c?????_members) > 2
        k_splits = Any[nothing for _=1:n_split_proposals]
        k_split_weights = [NaN for _=1:n_split_proposals]
        for j=1:n_split_proposals
            k_splits[j], k_split_lp = propose_split(tr?????, c?????, i)
            (_, tr?????, _) = k_splits[j]
            @assert !isnothing(tr?????.assignments[i])
            log_k_pdf += k_split_lp
            k_split_weights[j] = delta_log_P(tr?????, tr?????) + h_assess(i, k_splits[j]...) - k_split_lp
        end
        log_ksplit_mean_weight = logsumexp(k_split_weights) .- log(n_split_proposals)
        # @assert log_ksplit_mean_weight ??? only(k_split_weights)

        logprobs = k_split_weights .- logsumexp(k_split_weights)
        k_split_idx = rand(Categorical(exp.(logprobs)))
        log_k_pdf += logprobs[k_split_idx]
        # @assert k_split_idx == 1
        k_split = k_splits[k_split_idx]
        (_, tr?????, _) = k_split
        @assert !isnothing(tr?????.assignments[i])
    else
        k_splits, k_split, tr?????, k_split_idx, k_split_weights = nothing, nothing, nothing, nothing, nothing
        log_ksplit_mean_weight = -Inf
    end
    # if length(c?????_members) > 2
    #     k_split, k_split_lp = propose_split(tr?????, c?????, i)
    #     (_, tr?????, _) = k_split
    #     log_k_pdf += k_split_lp
    #     log_k_split_weight = delta_log_P(tr?????, tr?????) + h_assess(i, k_split...) - k_split_lp
    # else
    #     k_split, tr????? = nothing, nothing
    #     log_k_split_weight = -Inf
    # end
    ordered_cluster_ids = collect(keys(tr?????.clusters))
    merge_scores = enumerate_merges(tr?????, c?????) # Dict mapping merge ids to delta P's.
    log_stay_score = 0.
    move, move_logscore = choosemove_and_score(log_stay_score, log_ksplit_mean_weight, merge_scores, ordered_cluster_ids)
    log_k_pdf += move_logscore

    tr????? = do_move_get_new_trace(tr?????, c?????, move, tr?????, ordered_cluster_ids)
    c????? = tr?????.assignments[i]

    # Sample splitting randomness for L:
    if move > 2 # merge
        # we must have initially put the datapoint into a cluster with >= 2 datapoints, then merged into another cluster with at least 1
        @assert length(tr?????.clusters[c?????]) >= 3
        merged_with_cluster = ordered_cluster_ids[move - 2]

        # Given that L made a split from the new merged cluster to
        # tr?????????[c?????] and tr?????????[merged_with_cluster],
        # what was L's split aux randomness?
        # (L splits to tr????? \ {i}, which is the same as tr?????????.)
        ul, log_h_ul = h_propose(i, tr?????????, Set([c?????, merged_with_cluster]))
        
        log_k_pdf += log_h_ul
        l_split = (ul, tr?????????, Set([c?????, merged_with_cluster]))
    else
        if length(tr?????.clusters[c?????]) > 2
            unincorporate_point!(tr?????, i)
            l_split, l_split_lp = propose_split(tr?????, c?????, i)
            log_k_pdf += l_split_lp
            incorporate_point!(tr?????, i, c?????)
        else
            l_split = nothing
        end
    end

    values_for_L = (; k_split, l_split, move, tr?????????, c?????, k_splits, k_split_idx, k_split_weights)

    return (tr?????, values_for_L, log_k_pdf)
end

function l_assess(tr?????, i, k, n_split_proposals)
    tr????? = copytrace(tr?????)
    log_l_pdf = 0.0
    c????? = tr?????.assignments[i]

    length(tr?????.clusters[c?????]) == 1 && return log_l_pdf
    k.move > 2 && @assert length(tr?????.clusters[c?????]) >= 2

    unincorporate_point!(tr?????, i)

    if length(tr?????.clusters[c?????]) > 1
        lsplit_lp = assess_split(tr?????, c?????, i, k.l_split)
        (_, post_lsplit_tr, _) = k.l_split
        log_l_pdf += lsplit_lp
        log_l_split_weight = delta_log_P(tr?????, post_lsplit_tr) + h_assess(i, k.l_split...) - lsplit_lp
    else
        @assert k.move <= 2 "if L can't do a split, K cannot have done a merge."
        post_lsplit_tr = nothing
        log_l_split_weight = -Inf
    end
    merge_scores = enumerate_merges(tr?????, c?????)
    log_stay_score = 0.
    ordered_cluster_ids = collect(keys(tr?????.clusters))
    move = opposite_move(k.move, k.k_split, i, ordered_cluster_ids)
    log_l_pdf += score_move(log_stay_score, log_l_split_weight, merge_scores, ordered_cluster_ids, move)

    # tr????????? = do_move_get_new_trace(tr?????, c?????, move, l_split_trace, ordered_cluster_ids)
    # this should yield
    tr????????? = k.tr?????????
    # but use the one from `k` so that the cluster IDs are correct

    # If L did a split, K did a merge.  We have to assess the probability that before K merged,
    # it had put the new datapoint into the particular pre-merge cluster it did, rather than the other pre-merge cluster.
    if move == 2
        (_, _, l_split_cluster_ids) = k.l_split
        @assert k.c????? in l_split_cluster_ids
        other_cluster = only(c for c in l_split_cluster_ids if c != k.c?????)
        log_l_pdf += log_gibbs_probability_point_goes_in_first_cluster(tr?????????, k.c?????, other_cluster, i)
    end

    # If K put the new point into a cluster which had at least than 2 datapoints to start with,
    # K considered doing a split move.  Generate the randomness from that.
    if haskey(tr?????????.clusters, k.c?????) && length(tr?????????.clusters[k.c?????]) >= 2
        # First generate randomness for the non-chosen split moves.
        tr_prev = copytrace(tr?????????)
        incorporate_point!(tr?????????, i, k.c?????)
        for j=1:n_split_proposals
            j == k.k_split_idx && continue
            log_l_pdf += assess_split(tr?????????, k.c?????, i, k.k_splits[j])
        end
        unincorporate_point!(tr?????????, i)
        assert_is_equal(tr?????????, tr_prev)

        # Then generate randomness for the chosen split move.
        if move > 2
            # If a split move was implemented, we know what the split was, except the auxiliary variables.
            # Generate the auxiliary variables for the given split move, and pick a random index for the
            # chosen split.
            log_l_pdf += h_assess(i, k.k_split...)
            log_l_pdf -= log(n_split_proposals)
        else
            # If no split was implemented, generate a final split proposal,
            # and sample one of them according to the importance weight.
            # [For convenience I've written this to generate the chosen split last;
            # the true L kernel generates the chosen split at index k.k_split_idx.
            # But this shouldn't make a difference for the scoring.]
            incorporate_point!(tr?????????, i, k.c?????)
            log_l_pdf += assess_split(tr?????????, k.c?????, i, k.k_split)
            unincorporate_point!(tr?????????, i)
            log_l_pdf += (k.k_split_weights .- logsumexp(k.k_split_weights))[k.k_split_idx]
        end
    end

    incorporate_point!(tr?????, i, c?????)
    return log_l_pdf
end

### Helper functions ###
delta_log_P(trace, new_trace) = log_joint(new_trace) - log_joint(trace)

score_move(log_stay_score, log_k_split_weight, merge_scores, ordered_merge_ids, move) =
    choosemove_and_score(log_stay_score, log_k_split_weight, merge_scores, ordered_merge_ids; move)[2]
function choosemove_and_score(log_stay_score, log_k_split_weight, merge_scores, ordered_merge_ids; move=nothing)
    log_merge_pdfs = [merge_scores[id] for id in ordered_merge_ids]
    unnormalized_logprobs = vcat([log_stay_score, log_k_split_weight], log_merge_pdfs)
    normalized_logprobs = unnormalized_logprobs .- logsumexp(unnormalized_logprobs)
    if isnothing(move)
        probs = exp.(normalized_logprobs)
        move = rand(Categorical(probs))
    end
    return (move, normalized_logprobs[move])
end

function do_move_get_new_trace(trace, source_cluster_id, move, split_trace, ordered_merge_ids)::DPMMTrace
    if move == 1 # stay
        return trace
    elseif move == 2 # split
        return split_trace
    else
        cluster_to_merge_with = ordered_merge_ids[move - 2]
        newtrace = copytrace(trace)
        merge_clusters!(newtrace, source_cluster_id, cluster_to_merge_with)
        return newtrace
    end
end
function merge_clusters!(trace, source_cluster_id, other_cluster_id)
    points_in_other_cluster = copy(trace.clusters[other_cluster_id].members)
    for point in points_in_other_cluster
        unincorporate_point!(trace, point)
        incorporate_point!(trace, point, source_cluster_id)
    end
end

# Assumes `i` is not incorporated into any cluster in `trace`.
function log_gibbs_probability_point_goes_in_first_cluster(trace, c1, c2, i)
    new_cluster = singleton_cluster(trace, i)
    lp1 = crp_log_prior_predictive(trace, c1, 1) + conditional_likelihood(trace, c1, new_cluster)
    lp2 = crp_log_prior_predictive(trace, c2, 1) + conditional_likelihood(trace, c2, new_cluster)
    logprob = lp1 - logsumexp([lp1, lp2])
    return logprob
end

function opposite_move(move, k_split, i, ordered_cluster_ids)
    if move == 1 # stay -> stay
        return 1
    elseif move > 2 # merge -> split
        return 2
    else # split -> Merge
        @assert move == 2
        # this turns into a merge.  Q: which one?
        # A: merge with the cluster `k_split` outputted that doesn't contain the new datapoint
        (_, post_ksplit_trace, cluster_ids_to_merge) = k_split
        datapoint_cluster = post_ksplit_trace.assignments[i]
        other_ids = [id for id in cluster_ids_to_merge if id != datapoint_cluster]
        @assert length(other_ids) == 1 "other_ids = $other_ids ; datapoint_cluster = $datapoint_cluster"
        other_cluster = only(other_ids)
        return 2 + findfirst(x -> x == other_cluster, ordered_cluster_ids)
    end
end
