# Training execution preparation

The training worker exposes one framework-neutral preparation seam,
`TrainStepExecutor.prepare_execution(inputs, targets)`. A backend may use this
seam to prepare its executable for the first input signature before the first
batch is executed.

The preparation contract is strict:

- it may lower or compile backend kernels for the supplied signatures;
- it must not mutate model, optimizer, gradient-accumulation, or step state;
- it must not apply an optimizer update, consume source work, acknowledge a
  lease, or execute a dummy batch;
- the first real batch remains the only payload passed to `micro_step`.

The distributed JAX executor prepares the first micro-step, accumulation, and
macro-update kernels using the actual input and target signatures. It passes
abstract shapes, dtypes, and shardings to the compiler, so preparation does not
materialize batch-sized copies or zero-gradient buffers on host or device.
Preparation is cached by key, shape, and dtype, so a repeated signature does not
trigger a second compile. A backend that has no separate preparation work
implements the same seam as a no-op. Fluxway consumes only the executor contract
and does not know about JAX, XLA, or any backend-specific compilation
mechanism.

Compilation failures propagate to the owning operator. The worker does not
retry, sleep, substitute a smaller batch, or silently execute an unprepared
path.

When execution tracing is enabled, the adapter records
`operator_prepare_begin/end/error` and `operator_invocation_begin/end/error`
with the step, rank, SPMD identifier, and a monotonic sequence. These records
make a long first compilation distinguishable from a later invocation or a
coordination failure without changing Fluxway scheduling semantics.
