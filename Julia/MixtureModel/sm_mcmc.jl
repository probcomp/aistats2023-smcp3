using Distributions: DiscreteUniform

# global acc_ct = 0
# global rej_ct = 0
sm_mcmc!(trace) = sm_mcmc!(trace, maximum(trace.active))
function sm_mcmc!(trace, i)
    (new_tr, fwd_prob, details) = mcmc_propose(trace, i)
    bwd_prob = mcmc_assess(new_tr, i, trace, details)
    log_score = log_joint(new_tr) - log_joint(trace) + bwd_prob - fwd_prob
    acc_prob = min(1, exp(log_score))

    # Logging 
    # if mod(acc_ct + rej_ct, 1000) == 0
    #     println("""
    #     Accepted: $acc_ct
    #     Rejected: $rej_ct
    #     """)
    # end

    if rand() < acc_prob
        # println("accept")
        # global acc_ct += 1
        return new_tr
    else
        # global rej_ct += 1
        return trace
    end
end

function mcmc_propose(trace, i)
    if i == 1
        return (trace, 0., nothing)
    end
    logpdf = 0.

    p1 = rand(DiscreteUniform(1, i))
    p2 = rand(Categorical([j == p1 ? 0. : 1.0/(i-1.) for j=1:i]))
    logpdf -= log(i)
    logpdf -= log(i-1)

    c1, c2 = trace.assignments[p1], trace.assignments[p2]
    if c1 != c2 # merge
        trace = copytrace(trace)
        merge_clusters!(trace, c1, c2)
    else # split
        # println("split $c1 $c2")
        old_trace = trace
        (trace, split_logpdf) = do_split_using_chaperones(trace, p1, p2, nothing, gensym(), gensym(), collect(trace.clusters[c1].members), nothing)
        logpdf += split_logpdf
        @assert length(trace.clusters) > length(old_trace.clusters) "$(length(trace.clusters)), $(length(old_trace.clusters))"
    end

    return (trace, logpdf, (p1, p2))
end
function mcmc_assess(old_trace, i, new_trace, move_details)
    if i == 1
        return 0.
    end
    (p1, p2) = move_details
    c1, c2 = old_trace.assignments[p1], old_trace.assignments[p2]
    logpdf = 0.
    logpdf -= log(i)
    logpdf -= log(i-1)

    if c1 == c2
        c1_, c2_ = new_trace.assignments[p1], new_trace.assignments[p2]
        logpdf += do_split_using_chaperones(old_trace, p1, p2, nothing, c1_, c2_, collect(old_trace.clusters[c1].members), new_trace)[2]
    end

    return logpdf
end