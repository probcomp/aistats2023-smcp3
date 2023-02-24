## Renderer-Based Tracking Using SMCP3
- Instantiate the package before trying to run any of the scripts.
- The script `run_notebook.sh` launches a Jupyter notebook that can generate
  Meshcat visualizations of the model and the inference algorithms.
- The script `run_experiments.sh` runs the experiments for inference in a model
  with a renderer-based likelihood. Make sure a directory named `data` exists.
- The plot in the paper is created by using `src/make_plot.jl`. The value of
  the estimates of log marginal likelihood should be extracted from the output
  of `./run_experiments.sh`, and put in the `results` and `times` arrays, with
  the first element of each array being the estimates from SMCP3 and the second
  element being the estimates from bootstrap particle filtering. You can just
  copy the last two lines of the output of `run_experiments.sh` (the `times`
  and `results`) array and paste them in `src/make_plot.jl`.
