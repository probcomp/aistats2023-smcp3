function sort_by_minidx(Πₜ)
    v = collect(Πₜ)
    min_indices = map(minimum, v)
    perm = sortperm(min_indices)
    return v[perm]
end

function add_to_cluster(sorted_partition::Vector, cluster_idx::Int, y)
    return Set(
        i == cluster_idx ? union(set, Set([y])) : set
        for (i, set) in enumerate(sorted_partition)
    )
end
add_to_cluster(Π::Set, cluster::Set, y) =
    Set(s == cluster ? union(s, Set([y])) : s for s in Π)

function add_singleton(Π, y)
    new_pi =  union(Π, Set([Set([y])]))
    return new_pi
end
"""Remove index `y` from its cluster in Π"""
function remove(Π, y)
    return remove_empties(Set(
        y ∈ s ? setdiff(s, Set([y])) : s
        for s in Π
    ))
end
remove_empties(Π) = setdiff(Π, Set([Set()]))
find_sorted(Π, v) = find(sort_by_minidx(Π), v)
function find(Π, y::Int)
    for (i, s) in enumerate(Π)
        if y in s
            return i
        end
    end
    error("$y not in any set in $Π")
end
function find(Π, y::Set)
    for (i, s) in enumerate(Π)
        if y == s
            return i
        end
    end
    error("$y not in $Π")
end
cluster_containing(Π, i) = only(s for s in Π if i in s)
function merge(Π, s, c)
    union(
        setdiff(Π, Set([s, c])), Set([union(s, c)])
    )
end
function post_split_clusters(Π, merged)
    clusters = Set([s for s in Π if any(a in s for a in merged)])
    @assert union(clusters...) ⊆ merged "clusters = $clusters ; merged = $merged"
    return clusters
end