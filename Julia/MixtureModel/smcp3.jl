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
1. trₜ₋₁ : Previous trace, without the new datapoint.
2. trₜ¹  : (1) + the new datapoint is incorporated into a cluster.
3. trₜˢ  : (2) + the cluster containing the new datapoint has been split.
4. trₜ²  : (1) + either a split move, merge move, or stay move has been performed.

Datapoint indices:
1. i : index of new datapoint to incorporate

Cluster IDs:
1. cᵢ¹ : ID of the cluster `i` was incorporated into in trₜ¹
=#
function k_propose(trₜ₋₁, i, n_split_proposals)
    log_k_pdf = 0.0

    trₜ¹ = copytrace(trₜ₋₁)
    log_k_pdf += locally_optimal_smc_step!(trₜ¹, i; get_proposal_probability = true)
    
    cᵢ¹ = trₜ¹.assignments[i]
    cᵢ¹_members = trₜ¹.clusters[cᵢ¹].members
    if length(cᵢ¹_members) == 1
        # println("returning after smc step.")
        # println("after smc step, log_k_pdf is $log_k_pdf")
        return (trₜ¹, nothing, log_k_pdf)
    end

    if length(cᵢ¹_members) > 2
        k_splits = Any[nothing for _=1:n_split_proposals]
        k_split_weights = [NaN for _=1:n_split_proposals]
        for j=1:n_split_proposals
            k_splits[j], k_split_lp = propose_split(trₜ¹, cᵢ¹, i)
            (_, trₜˢ, _) = k_splits[j]
            @assert !isnothing(trₜˢ.assignments[i])
            log_k_pdf += k_split_lp
            k_split_weights[j] = delta_log_P(trₜ¹, trₜˢ) + h_assess(i, k_splits[j]...) - k_split_lp
        end
        log_ksplit_mean_weight = logsumexp(k_split_weights) .- log(n_split_proposals)
        # @assert log_ksplit_mean_weight ≈ only(k_split_weights)

        logprobs = k_split_weights .- logsumexp(k_split_weights)
        k_split_idx = rand(Categorical(exp.(logprobs)))
        log_k_pdf += logprobs[k_split_idx]
        # @assert k_split_idx == 1
        k_split = k_splits[k_split_idx]
        (_, trₜˢ, _) = k_split
        @assert !isnothing(trₜˢ.assignments[i])
    else
        k_splits, k_split, trₜˢ, k_split_idx, k_split_weights = nothing, nothing, nothing, nothing, nothing
        log_ksplit_mean_weight = -Inf
    end
    # if length(cᵢ¹_members) > 2
    #     k_split, k_split_lp = propose_split(trₜ¹, cᵢ¹, i)
    #     (_, trₜˢ, _) = k_split
    #     log_k_pdf += k_split_lp
    #     log_k_split_weight = delta_log_P(trₜ¹, trₜˢ) + h_assess(i, k_split...) - k_split_lp
    # else
    #     k_split, trₜˢ = nothing, nothing
    #     log_k_split_weight = -Inf
    # end
    ordered_cluster_ids = collect(keys(trₜ¹.clusters))
    merge_scores = enumerate_merges(trₜ¹, cᵢ¹) # Dict mapping merge ids to delta P's.
    log_stay_score = 0.
    move, move_logscore = choosemove_and_score(log_stay_score, log_ksplit_mean_weight, merge_scores, ordered_cluster_ids)
    log_k_pdf += move_logscore

    trₜ² = do_move_get_new_trace(trₜ¹, cᵢ¹, move, trₜˢ, ordered_cluster_ids)
    cᵢ² = trₜ².assignments[i]

    # Sample splitting randomness for L:
    if move > 2 # merge
        # we must have initially put the datapoint into a cluster with >= 2 datapoints, then merged into another cluster with at least 1
        @assert length(trₜ².clusters[cᵢ²]) >= 3
        merged_with_cluster = ordered_cluster_ids[move - 2]

        # Given that L made a split from the new merged cluster to
        # trₜ₋₁[cᵢ¹] and trₜ₋₁[merged_with_cluster],
        # what was L's split aux randomness?
        # (L splits to trₜ¹ \ {i}, which is the same as trₜ₋₁.)
        ul, log_h_ul = h_propose(i, trₜ₋₁, Set([cᵢ¹, merged_with_cluster]))
        
        log_k_pdf += log_h_ul
        l_split = (ul, trₜ₋₁, Set([cᵢ¹, merged_with_cluster]))
    else
        if length(trₜ².clusters[cᵢ²]) > 2
            unincorporate_point!(trₜ², i)
            l_split, l_split_lp = propose_split(trₜ², cᵢ², i)
            log_k_pdf += l_split_lp
            incorporate_point!(trₜ², i, cᵢ²)
        else
            l_split = nothing
        end
    end

    values_for_L = (; k_split, l_split, move, trₜ₋₁, cᵢ¹, k_splits, k_split_idx, k_split_weights)

    return (trₜ², values_for_L, log_k_pdf)
end

function l_assess(trₜ², i, k, n_split_proposals)
    trₜ² = copytrace(trₜ²)
    log_l_pdf = 0.0
    cᵢ² = trₜ².assignments[i]

    length(trₜ².clusters[cᵢ²]) == 1 && return log_l_pdf
    k.move > 2 && @assert length(trₜ².clusters[cᵢ²]) >= 2

    unincorporate_point!(trₜ², i)

    if length(trₜ².clusters[cᵢ²]) > 1
        lsplit_lp = assess_split(trₜ², cᵢ², i, k.l_split)
        (_, post_lsplit_tr, _) = k.l_split
        log_l_pdf += lsplit_lp
        log_l_split_weight = delta_log_P(trₜ², post_lsplit_tr) + h_assess(i, k.l_split...) - lsplit_lp
    else
        @assert k.move <= 2 "if L can't do a split, K cannot have done a merge."
        post_lsplit_tr = nothing
        log_l_split_weight = -Inf
    end
    merge_scores = enumerate_merges(trₜ², cᵢ²)
    log_stay_score = 0.
    ordered_cluster_ids = collect(keys(trₜ².clusters))
    move = opposite_move(k.move, k.k_split, i, ordered_cluster_ids)
    log_l_pdf += score_move(log_stay_score, log_l_split_weight, merge_scores, ordered_cluster_ids, move)

    # trₜ₋₁ = do_move_get_new_trace(trₜ², cᵢ², move, l_split_trace, ordered_cluster_ids)
    # this should yield
    trₜ₋₁ = k.trₜ₋₁
    # but use the one from `k` so that the cluster IDs are correct

    # If L did a split, K did a merge.  We have to assess the probability that before K merged,
    # it had put the new datapoint into the particular pre-merge cluster it did, rather than the other pre-merge cluster.
    if move == 2
        (_, _, l_split_cluster_ids) = k.l_split
        @assert k.cᵢ¹ in l_split_cluster_ids
        other_cluster = only(c for c in l_split_cluster_ids if c != k.cᵢ¹)
        log_l_pdf += log_gibbs_probability_point_goes_in_first_cluster(trₜ₋₁, k.cᵢ¹, other_cluster, i)
    end

    # If K put the new point into a cluster which had at least than 2 datapoints to start with,
    # K considered doing a split move.  Generate the randomness from that.
    if haskey(trₜ₋₁.clusters, k.cᵢ¹) && length(trₜ₋₁.clusters[k.cᵢ¹]) >= 2
        # First generate randomness for the non-chosen split moves.
        tr_prev = copytrace(trₜ₋₁)
        incorporate_point!(trₜ₋₁, i, k.cᵢ¹)
        for j=1:n_split_proposals
            j == k.k_split_idx && continue
            log_l_pdf += assess_split(trₜ₋₁, k.cᵢ¹, i, k.k_splits[j])
        end
        unincorporate_point!(trₜ₋₁, i)
        assert_is_equal(trₜ₋₁, tr_prev)

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
            incorporate_point!(trₜ₋₁, i, k.cᵢ¹)
            log_l_pdf += assess_split(trₜ₋₁, k.cᵢ¹, i, k.k_split)
            unincorporate_point!(trₜ₋₁, i)
            log_l_pdf += (k.k_split_weights .- logsumexp(k.k_split_weights))[k.k_split_idx]
        end
    end

    incorporate_point!(trₜ², i, cᵢ²)
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
