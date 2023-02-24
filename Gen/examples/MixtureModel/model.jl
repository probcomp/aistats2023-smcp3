import Random
include("distributions/crp.jl")
include("distributions/bernoulli_logscale.jl")
function get_dpmm(alpha, cluster_distribution, cluster_hyperparams)
    @gen function dpmm(records)
        N = length(records)
        partition ~ crp(N, alpha)

        logprob = 0.0
        for table in partition
            generated_bag = Set(records[i] for i in table)
            logprob += Gen.logpdf(cluster_distribution, generated_bag, length(table), cluster_hyperparams)
        end
        # To unnormalize this distribution, condition on this being true.
        condition_this ~ bernoulli_logscale(logprob)
    end
end

function generate_data(N, alpha, cluster_distribution::Distribution{Set{T}}, cluster_hyperparams) where {T}
    output = Array{T}(undef, N)
    partition = random(crp, N, alpha)
    for indices in partition
        values = random(cluster_distribution, length(indices), cluster_hyperparams)
        ordered_indices = Random.shuffle(collect(indices))
        ordered_values = Random.shuffle(collect(values))
        for (idx, val) in zip(ordered_indices, ordered_values)
            output[idx] = val
        end
    end
    return output
end