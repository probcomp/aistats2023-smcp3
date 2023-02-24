using StatsFuns: loggamma

#=
FSGaussian = Fixed Standard-Deviation Gaussian.
=#

function generate_synthetic_fsgaussian_trace(hypers, alpha, n; sorted=false)
    partition = simulate_crp(alpha, n)
    mus   = [rand(Normal(hypers.μ₀, hypers.σ₀)) for _ in partition]
    data  = vcat([[rand(Normal(mu, hypers.σ)) for i in 1:n] for (n, mu) in zip(partition, mus)]...) 
    perm = sorted ? sortperm(data) : shuffle(1:n)
    data = data[(perm)]
    trace = create_initial_dpmm_trace(hypers, FSGaussianCluster, data, alpha)
    ctr = 1
    for n in partition
        c = gensym()
        for i=ctr:(ctr+n-1)
            incorporate_point!(trace, invperm(perm)[i], c)
        end
        ctr += n
    end
    return trace
end

struct FSGaussianHyperparameters <: Hyperparameters
    μ₀ :: Float64 # mean of cluster mean prior
    σ₀ :: Float64 # variance of cluster mean prior
    σ  :: Float64 # cluster variance
end

mutable struct FSGaussianStatistics <: ClusterStats
    sum     :: Float64
    sq_sum  :: Float64
    n   :: Int
end

# From https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf, section 2.5
function log_marginal_data_prob(s::FSGaussianStatistics, h::FSGaussianHyperparameters)
    a = log(h.σ) - s.n*log(√((2 * π * h.σ))) - log(√(s.n * h.σ₀^2 + h.σ^2))
    b = -s.sq_sum/(2 * h.σ^2) - h.μ₀/(2 * h.σ₀^2)
    c = (
        (h.σ₀^2 * s.sum^2 / h.σ^2)
        + (h.σ^2 * h.μ₀^2 / h.σ₀^2)
        + 2*s.sum*h.μ₀
    ) / (
        2 * (s.n * h.σ₀^2 + h.σ^2)
    )
    return a + b + c
end

function Base.isequal(a::FSGaussianStatistics, b::FSGaussianStatistics)
    a.sum == b.sum && a.sq_sum == b.sq_sum && a.n == b.n
end
function Base.isapprox(a::FSGaussianStatistics, b::FSGaussianStatistics)
    isapprox(a.sum, b.sum, atol=1e-4) && isapprox(a.sq_sum, b.sq_sum, atol=1e-4) && a.n == b.n
end

copystats(stats::FSGaussianStatistics) = FSGaussianStatistics(stats.sum, stats.sq_sum, stats.n)

const FSGaussianCluster = Cluster{FSGaussianStatistics, FSGaussianHyperparameters}

const FSGaussianTrace = DPMMTrace{FSGaussianCluster, FSGaussianHyperparameters}

# Initializing cluster stats

function empty_cluster_stats(trace::FSGaussianTrace)
    FSGaussianStatistics(0.0, 0.0, 0)
end

# Updating sufficient statistics

function update_stats_after_adding_point!(cluster::FSGaussianCluster, x::Float64)
    cluster.stats.sum += x
    cluster.stats.sq_sum += x^2
    cluster.stats.n += 1
end

function update_stats_after_removing_point!(cluster::FSGaussianCluster, x::Float64)
    cluster.stats.sum -= x
    cluster.stats.sq_sum -= x^2
    cluster.stats.n -= 1
end

function singleton_cluster(trace::FSGaussianTrace, index)
    x = trace.data[index]
    FSGaussianCluster(trace, Set(index), FSGaussianStatistics(x, x^2, 1))
end

# If cluster_id is new, gives the marginal likelihood.
function conditional_likelihood(trace::FSGaussianTrace, cluster_id::Symbol, provisional_cluster::FSGaussianCluster)
    existing_cluster = haskey(trace.clusters, cluster_id) ? trace.clusters[cluster_id] : Cluster(trace, Set{Int}(), empty_cluster_stats(trace))
    
    return (
        log_marginal_data_prob(
            merge_stats(existing_cluster.stats, provisional_cluster.stats), trace.hypers
        ) - log_marginal_data_prob(existing_cluster.stats, trace.hypers)
    )
end

# Update stats of existing cluster to include new datapoints
function merge_stats!(trace::FSGaussianTrace, existing_cluster_id::Symbol, new_cluster::FSGaussianCluster)
    cluster = trace.clusters[existing_cluster_id]
    new_st = merge_stats(cluster.stats, new_cluster.stats)
    cluster.stats.sum = new_st.sum
    cluster.stats.sq_sum = new_st.sq_sum
    cluster.stats.n = new_st.n
end
function merge_stats(a::FSGaussianStatistics, b::FSGaussianStatistics)
    FSGaussianStatistics(a.sum + b.sum, a.sq_sum + b.sq_sum, a.n + b.n)
end

# function summarize(trace::GaussianTrace)
#     [(length(cluster.members), sort(collect(cluster.members)), 
#      cluster.stats.data_mean, 
#      sqrt(cluster.stats.data_sse / length(cluster.members))) 
#      for (_, cluster) in trace.clusters]
# end