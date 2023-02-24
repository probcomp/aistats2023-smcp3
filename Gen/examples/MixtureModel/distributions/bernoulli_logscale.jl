struct BernoulliLogscale <: Distribution{Bool} end

"""
    bernoulli_logscale(logprob::Real)

Sample a Boolean from `Bern(exp(logprob))`.
"""
const bernoulli_logscale = BernoulliLogscale()

function Gen.logpdf(::BernoulliLogscale, x::Bool, logprob::Real)
    if x
        return logprob
    else
        return log(1 - exp(logprob))
    end
end

function Gen.random(::BernoulliLogscale, logprob::Real)
    Gen.random(bernoulli, exp(logprob))
end

is_discrete(::BernoulliLogscale) = true

(::BernoulliLogscale)(logprob) = random(BernoulliLogscale(), logprob)

Gen.has_output_grad(::BernoulliLogscale) = false

# Could be true; would need to add implementatation for logpdf_grad.
Gen.has_argument_grads(::BernoulliLogscale) = (false,)
