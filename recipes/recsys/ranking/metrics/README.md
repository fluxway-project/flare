# Training metric throughput

Recommendation training uses a shared metric worker. Its default behavior
waits for Ray Data reduction and metric writing at each window boundary. The
following options are available on the training CLIs:

- `--metric-ray-parallelism N` sets the Ray Data CPU partition limit (default 2).
  It does not add metric SPMD ranks or reserve N dedicated CPUs. More partitions
  help only when the Ray Data computation has enough available CPU and data.
- `--metric-max-pending-windows N` submits completed windows in the background
  when N is positive. Zero (the default) keeps synchronous window completion.
  The bound deliberately applies backpressure when N windows remain pending.
  Results are written in window order, and terminal steps or shutdown wait for
  all pending results and surface errors.
- `--metric-sample-interval-steps N` controls how often scores and loss are
  copied from GPU to host (default 1). Increasing it reduces collection work
  but changes the statistical sample used for AUC/GAUC.

No finite resource allocation can guarantee zero training impact if metric
production outruns metric processing indefinitely. Even with background windows,
sampled steps still perform a blocking GPU-to-host transfer inside the training
worker, and the metric worker builds an Arrow table and calls `ray.put`. Measure
training throughput, `materialize_train_metrics` time, Ray Data window latency,
and pending-window occupancy before choosing the settings.
