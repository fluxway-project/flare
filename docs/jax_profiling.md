# Profiling JAX Training

This document provides guidelines and workflows for profiling JAX distributed training workloads in Flare-JAX.

## Tracing Training Steps

To profile JAX models accurately, capture compiled execution after warmup. Throughput measurements must be separated from the tracing window. Always fence device execution before and after tracing steps:

```python
import jax

# Warmup to trigger compilation and memory initialization
for _ in range(warmup_steps):
    state, outputs = train_step(state, batch)
jax.block_until_ready((state, outputs))

# Configure profiler options
options = jax.profiler.ProfileOptions()
options.host_tracer_level = 2
options.python_tracer_level = 0

jax.profiler.start_trace(
    "./jax-trace",
    profiler_options=options,
    create_perfetto_trace=True,
)
try:
    for step in range(2):
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            state, outputs = train_step(state, batch)
            jax.block_until_ready((state, outputs))
finally:
    jax.profiler.stop_trace()
```

### Distributed Tracing Rules
- Assign a distinct trace directory for each rank or selected worker process (e.g., `./jax-trace/rank_${RANK}`).
- Execute identical collective sequences across all ranks to prevent collective desynchronization or deadlocks.
- Synchronize across processes using collective barriers around the profiling window.
- Never include profiler startup or shutdown overhead in timed execution benchmarks.

## Trace Analysis and Summarization

Traces can be visualized in Perfetto UI (`ui.perfetto.dev`) or analyzed using TensorBoard XProf.

To summarize GPU busy time and kernel distributions for a single-GPU trace directory:

```bash
python tools/validation/summarize_jax_profile.py ./jax-trace \
  --step-name train_step \
  --output ./jax-profile-summary.json
```

Key considerations when interpreting summaries:
- **GPU Busy Time**: Calculated via interval unions to avoid double-counting overlapping streams.
- **Kernel Concurrency**: Category durations may overlap across CUDA streams.
- **NCCL Overhead**: Collective operation durations include synchronization and rank waiting time, not solely raw bus transmission time.
- **Data Privacy**: Traces capture hostnames, environment paths, tensor shapes, and module names. Sanitize trace files before sharing externally.

## Memory Allocator Configuration

Compare memory allocators on identical models, batch sizes, accumulation counts, and precision settings. Environment variables must be set prior to importing JAX:

```bash
# Platform allocator (allocates via cudaMalloc/cudaFree, returns memory to OS)
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python train.py

# Default pool allocator (pre-allocates and manages BFC buffer pool)
XLA_PYTHON_CLIENT_ALLOCATOR=default \
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
  XLA_CLIENT_MEM_FRACTION=0.90 \
  python train.py
```

### Allocator Trade-offs
- **Platform Allocator**: Frees unused memory back to the driver. Reduces peak allocated memory at the cost of frequent driver synchronization overhead.
- **Default (BFC) Allocator**: Eliminates allocation stalls by reusing memory pools. In multi-tenant environments, adjust `XLA_CLIENT_MEM_FRACTION` to leave headroom for NCCL, CPU/GPU staging, and external runtimes.
- **cuda_async**: Available on supported CUDA versions; evaluate empirically before enabling in production.

## Critical Path Diagnosis

| Symptom | Primary Causes | Investigation Steps |
| :--- | :--- | :--- |
| **Large GPU Stream Gaps** | Host dispatch latency, CPU-GPU data transfers, recompilation | Verify input pipeline prefetching, host callback stalls, and buffer donations (`donate_argnums`). Check for dynamic shape recompilations. |
| **Dominant NCCL Collectives** | Load imbalance, stragglers, suboptimal network topology | Distinguish gradient reductions (`all-reduce`, `reduce-scatter`) from weight gathering (`all-gather`). Verify collective dtypes (e.g., BF16 transport) and NIC/NVLink topology. |
| **Compute Bottlenecks** | Suboptimal GEMMs, memory layout conversions, scatter/gather overhead | Inspect GEMM tile sizes and Tensor Core utilization. Check whether custom ops or layout transposes trigger memory bandwidth bottlenecks. |
| **Memory Growth & OOM** | Inefficient buffer donation, fragmentation | Inspect XLA compiled HLO memory analysis. Verify that donated buffers appear in `alias_size_in_bytes`. |

Optimize bottlenecks iteratively based on critical path proportions:
$$\text{Speedup} = \frac{1}{(1 - f) + \frac{f}{s}}$$
where $f$ is the fraction of total runtime spent in the optimized segment, and $s$ is the segment speedup factor.

## ZeRO Topology Benchmarking

Evaluate communication efficiency between global ZeRO partitioning and node-local ZeRO ownership using the validation benchmark:

```bash
# Global ZeRO group across nodes
python tools/validation/benchmark_zero_topology.py \
  --mode global \
  --nodes 2 \
  --gpus-per-node 8 \
  --elements 67108864 \
  --output /tmp/global-zero.json

# Node-local ZeRO with cross-node gradient reduction
python tools/validation/benchmark_zero_topology.py \
  --mode node_local \
  --nodes 2 \
  --gpus-per-node 8 \
  --elements 67108864 \
  --output /tmp/node-local-zero.json
```

> [!NOTE]
> The total number of elements must be divisible by the total GPU count. Communication speedups measured in isolated micro-benchmarks must always be validated against end-to-end `train_core` execution throughput.
