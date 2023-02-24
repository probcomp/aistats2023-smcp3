##################
# Noisy Renderer #
##################

using LinearAlgebra: I
using StatsFuns: logsumexp

struct RendererChoiceMap <: ChoiceMap
    noisy_pc::MMatrix
end

Gen.has_value(c::RendererChoiceMap, i::Int) = 1 <= i <= size(c.noisy_pc)[2]
Gen.get_value(c::RendererChoiceMap, i::Int) = c.noisy_pc[:, i]

get_values_shallow(c::RendererChoiceMap) = enumerate(eachcol(c.noisy_pc))

struct RendererTrace <: Trace
    noisy_pc::MMatrix
    pc::MMatrix
    args::Tuple{Scene,Camera,Int,<:Real}
    score::Float64
end

struct NoisyRenderer <: GenerativeFunction{MMatrix,RendererTrace} 
end

const noisy_renderer = NoisyRenderer()

Gen.get_args(tr::RendererTrace) = tr.args
Gen.get_retval(tr::RendererTrace) = tr.noisy_pc
Gen.get_choices(tr::RendererTrace) = RendererChoiceMap(tr.noisy_pc)
Gen.get_score(tr::RendererTrace) = tr.score
Gen.get_gen_fn(tr::RendererTrace) = noisy_renderer

Gen.project(::RendererTrace, ::EmptySelection) = 0.

_uniform_gm_logpdf(x, mus, covar) = 
    -log(size(mus)[2]) + logsumexp(map(i->logpdf(mvnormal, mus[:, i], x, covar),
                                       1:size(mus)[2]))

function Gen.simulate(::NoisyRenderer, args::Tuple{Scene,Camera,Int,<:Real})
    scene, camera, num_samples, noise = args
    pc = render_point_cloud(scene, camera)
    N, covar = size(pc)[2], noise * Matrix(I, 3, 3)
    noisy_pc, score = MMatrix{3,num_samples,Float64}(undef), 0
    for i=1:num_samples
        noisy_pc[:, i] = mvnormal(pc[:, uniform_discrete(1, N)], covar)
        score += _uniform_gm_logpdf(noisy_pc[:, i], pc, covar)
    end
    RendererTrace(noisy_pc, pc, args, score)
end

function Gen.generate(::NoisyRenderer, args::Tuple{Scene,Camera,Int,<:Real},
                      constraints::ChoiceMap)
    scene, camera, num_samples, noise = args
    pc = render_point_cloud(scene, camera)
    N, covar = size(pc)[2], noise * Matrix(I, 3, 3)
    noisy_pc, score, weight = MMatrix{3,num_samples,Float64}(undef), 0, 0
    for i=1:num_samples
        noisy_pc[:, i] = has_value(constraints, i) ?  constraints[i] :
            mvnormal(pc[:, uniform_discrete(1, N)], covar)
        sample_score = _uniform_gm_logpdf(noisy_pc[:, i], pc, covar)
        score += sample_score
        weight += has_value(constraints, i) ? sample_score : 0
    end
    RendererTrace(noisy_pc, pc, args, score), weight
end
