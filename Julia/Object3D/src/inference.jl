using DifferentiableRenderer: diff_render_point_cloud
using LinearAlgebra: norm, dot

obs_likelihood(y, obs) =
    assess(noisy_renderer, (make_scene(0., y, 0.), cam, num_noisy_samples,
                            noise), RendererChoiceMap(obs))[1]

function _maybe_resample!(weights; ess_thres=2)
    n = length(weights)
    ess = logsumexp(weights)^2 / logsumexp(log(2) .+ weights)
    if ess < ess_thres
        θ = exp.(weights .- logsumexp(weights))
        parents = [categorical(θ) for _=1:n]
        #fill(sum(weights)/n, n), parents
        fill(logsumexp(weights) -log(n), n), parents
    else
        weights, collect(1:n)
    end
end

function bootstrap_particle_filter(obs, num_particles::Int=5)
    T = length(obs)
    particles, weights = [fill(0., num_particles)], [fill(0., num_particles)]
    parent_idxs = []
    for t=1:T
        ext = map(p->normal(p, DRIFT_VARIANCE), particles[t])
        ext_weights = map((w, p)->w + obs_likelihood(p, obs[t]), weights[t], ext)
        new_weights, new_parent_idxs = _maybe_resample!(ext_weights)
        push!(particles, [ext[new_parent_idxs[i]] for i=1:num_particles])
        push!(weights, new_weights)
        push!(parent_idxs, new_parent_idxs)
    end
    particles, weights, parent_idxs
end

const GRID = collect(-1.0:0.25:1.0)
function proposal(particle, obs)
    aux = normal(particle, DRIFT_VARIANCE)
    s = map(y->obs_likelihood(y, obs) + logpdf(normal, y, particle, DRIFT_VARIANCE),
            aux .+ GRID)
    s = exp.(s .- logsumexp(s))
    j = categorical(s)
    prop = (aux .+ GRID)[j]
    r = map(y->logpdf(normal, y, particle, DRIFT_VARIANCE), prop .+ GRID)
    r = exp.(r .- logsumexp(r))
    i = categorical(r)
    weight = sum([obs_likelihood(prop, obs),
                  logpdf(normal, prop, particle, DRIFT_VARIANCE),
                  -log(s[j]) - logpdf(normal, aux, particle, DRIFT_VARIANCE),
                  log(r[i])])
    prop, weight
end

function grid_smcp3(obs, num_particles::Int=5)
    T = length(obs)
    particles, weights = [fill(0., num_particles)], [fill(0., num_particles)]
    parent_idxs = []
    for t=1:T
        ext, ext_dweights = zip(map(p->proposal(p, obs[t]), particles[t])...)
        ext_weights = map((w, dw)->w + dw, weights[t], ext_dweights)
        new_weights, new_parent_idxs = _maybe_resample!(ext_weights)
        push!(particles, [ext[new_parent_idxs[i]] for i=1:num_particles])
        push!(weights, new_weights)
        push!(parent_idxs, new_parent_idxs)
    end
    particles, weights, parent_idxs
end
