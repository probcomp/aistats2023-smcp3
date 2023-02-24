## Renderer-Based Tracking Using SMCP3
- The script `run_notebook.sh` launches a Jupyter notebook that generates
  Meshcat visualizations of the model and the inference algorithms.
- The script `run_experiments.sh` runs the experiments for inference in a model
  with a renderer-based likelihood.
- The plot in the paper is created by using `src/make_plot.jl`. The value of
  the estimates of log marginal likelihood should be extracted from the output
  of `./run_experiments.sh`, and put in the `results` tuple, with the first
  element of the tuple being the estimates from bootstrap particle filtering
  and the second element being the estimates from SMCP3.
