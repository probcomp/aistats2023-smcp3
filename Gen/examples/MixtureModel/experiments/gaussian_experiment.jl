using Gen, GenTraceKernelDSL, Revise
const TracedKernel = GenTraceKernelDSL.TracedKernel
Base.isapprox(a::Set{Set{Int}}, b::Set{Set{Int}}) = a == b
Base.isapprox(a::Set{Int}, b::Set{Int}) = a == b
Base.isapprox(::Nothing, ::Nothing) = true

include("../distributions/gaussian_bag.jl")
include("../model.jl")
include("../inference/inference.jl")

GAUSSIAN_HYPERS = GaussianHypers(0, 1 / 100, 1 / 2, 1 / 2)
data = generate_data(100, 1.0, gaussian_bag, GAUSSIAN_HYPERS)
@time (state_locopt = smc_locopt(get_dpmm(4.0, gaussian_bag, GAUSSIAN_HYPERS), data, 20))
println("Log Z estimate - LocOpt: $(get_logz_estimate(state_locopt))")
@time (state_smcp3 = smcp3(get_dpmm(4.0, gaussian_bag, GAUSSIAN_HYPERS), data, 20; n_split_proposals=1))
println("Log Z estimate - SMCP3: $(get_logz_estimate(state_smcp3))")

include("../visualize.jl")
visualize_trace(state_smcp3.traces[1])

#=
There is currently a bug in the Gen implementation of the proposal distribution, in the case n_split_proposals > 1.
Having n_split_proposals > 1 is necessary to consistently achieve good performance under these hyperparameters.
See the `Julia/` directory for a correct implementation of the full inference algorithm, and for inference
with other likelihoods and on real (rather than synthetic) data.
=#