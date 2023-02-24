# SMCP3 Code Submission

## Top-level directory structure
- `Julia/` contains the code used to run our experiments, not using the automated implementation of SMCP3 in Gen.
- `Gen/` contains code implementing the automated implementation of SMCP3 in Gen, and implementations of some of our examples models and inference programs using this Gen support for SMCP3.

## Pointers to scripts for reproducing the experiments in the paper

### Experiments for figures in the paper body
- Figure 3
  - Left: `Julia/StateEstimation/time_plot.jl`
  - Right: `Julia/MixtureModel/gaussian/gaussian_figure.jl`
- Figure 5: `Julia/StateEstimation/plot_particles.jl`
- Figure 6 plot: `Julia/Object3D/run_experiments.sh` generates the data (to run, call `./run_experiments.sh` from within `Julia/Object3D/`); `Julia/Object3D/make_plot.jl` is used to render the plot.
- Table 1
  - State-space model: `Julia/StateEstimation/100d_table.jl` 
  - Mixture model (Medicare data): `Julia/MixtureModel/string/string-experiment.jl`

### Experiments for figures in the Appendix
<!-- Figure 7: To add once Matin sends us his code.  -->
- Figure 8: `Julia/StateEstimation/100d_time_plot.jl` 
- Table 2: `Julia/MixtureModel/string/string-experiment.jl`

## Contents of the `Gen/` directory
- `examples/` contains implementations of the SMCP3 algortihms from Sec. 4.1 and 4.2 of the paper, using our implementation of automated SMCP3 support in Gen.
    - `examples/StateEstimation` contains an implementation of our inference algorithm for the 1D object tracking model from noisy position observations (sec. 4.1).
    - `examples/MixtureModel` contains an implemenation of our inference algorithm for DPMMs (sec. 4.2).
- `lib/` contains our implementation of automated SMCP3 support for Gen.
    - `lib/DynamicForwardDiff.jl` is a custom library for automatic differentiation on trace-shaped data.
    - `lib/GenTraceKernelDSL.jl` is the main library providing support for SMCP3.  This exposes the `@kernel` macro used for writing probabilistic programs implementing SMCP3 kernels, and adds a method to the `Gen.particle_filter_step!` function for performing an SMCP3 step using kernels written in this DSL.  (This library also provides support for involutive MCMC using proposal programs written in this same DSL.)  For our implementation of SMCP3, see `lib/GenTraceKernelDSL.jl/src/inference.jl`.

## Running this code
1. Download Julia.
2. Activate and instantiate the Julia package.  (From the Julia repl, type `]activate` then `]instantiate`.)
3. Run the Julia scripts in the package.