function simulate_crp(alpha, n)
    tables = [1]
    for i in 2:n
        probs = [tables..., alpha] ./ (i - 1 + alpha)
        next_table = rand(Categorical(probs))
        if next_table > length(tables)
            push!(tables, 1)
        else
            tables[next_table] += 1
        end
    end
    return tables
end


function run_experiment(experiment, n_iters)
    vals = [NaN for _=1:n_iters]
    for i=1:n_iters
        vals[i] = experiment()
    end
    return vals
end

resample(args) = maybe_resample(args..., Inf)[1]
