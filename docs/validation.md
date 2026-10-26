# Data and checkpoint validation

Run validation tools from the repository root in the same Python environment
as the application. The tools read saved artifacts; they do not launch a
training job, mutate a catalog or repair a checkpoint.

## Committed source cuts

```bash
python -m tools.validation.inspect_checkpoint_source_cut \
  "$CHECKPOINT_DIR" "$CUT" --expected-reader-workers "$READER_WORKERS"
```

The inspector requires a committed checkpoint transaction, matching manifest
digest, matching framework/SPMD cuts and one matching position per Reader
partition. This is an offline consistency check. It does not establish what
happens when a process crashes during publication.

## Dense model and optimizer state

```bash
env -u SYSTEM_ENV -u GRAVITINO_TOKEN JAX_PLATFORMS=cpu \
  python -m tools.validation.audit_dense_checkpoint \
  --checkpoint-root "$CHECKPOINT_DIR" --step "$CUT" \
  --expected-optimizer-step "$OPTIMIZER_UPDATES" \
  --state-definition-factory "$STATE_DEFINITION_FACTORY" \
  --factory-kwargs "$STATE_DEFINITION_KWARGS" \
  --spmd-id "$DENSE_SPMD_ID" --component-id "$DENSE_COMPONENT_ID" \
  --scratch "$RUN_DIR/dense-audit-scratch" \
  --output "$RUN_DIR/dense-audit.json"
```

The application supplies a `module:factory` returning
`NnxTrainingStateDefinition` and a JSON object of factory arguments. The tool
resolves a committed checkpoint with `CheckpointRepository` and restores its
dense component through `OrbaxCheckpointComponent` onto one CPU device. It
checks the expected optimizer update count, NNX optimizer step and finite
model/optimizer state, then reports state fingerprints and whether the
checkpoint file size/mtime inventory was preserved.

Use new scratch and output paths outside the protected checkpoint root. The
process must have `JAX_PLATFORMS=cpu` and no `SYSTEM_ENV` or `GRAVITINO_TOKEN`.
Calculate the expected update count from every accumulation window, including
partial windows flushed at checkpoint boundaries. A data microstep count is
not necessarily the optimizer update count. This audit covers dense state;
sparse-state and controlled-crash evidence remain separate.

## Native sparse state

```bash
env -u SYSTEM_ENV -u GRAVITINO_TOKEN JAX_PLATFORMS=cpu \
  python -m tools.validation.audit_sparse_checkpoint \
  --checkpoint-root "$CHECKPOINT_DIR" --step "$CUT" \
  --expected-optimizer-step "$OPTIMIZER_UPDATES" \
  --store-factory "$SPARSE_STORE_FACTORY" \
  --factory-kwargs "$SPARSE_STORE_KWARGS" \
  --output "$RUN_DIR/sparse-audit.json"
```

The application's `module:factory` returns a mapping from SPMD IDs to
`(component_id, create_optimizer_enabled_store)` pairs. Each zero-argument
constructor must create a store matching the saved
key/value/optimizer dtypes, dimensions, optimizer and score policy. The tool
restores each component through `SparseEmbeddingCheckpointComponent` to one
CPU partition and reads the native structured `export_page_images` interface.
It creates and releases stores one component at a time. It checks per-page
shape alignment, unique keys within each dimension, complete store coverage, finite
values and optimizer state, and per-key update counts bounded by the supplied
optimizer count. It reports fingerprints in key order and preserves the
checkpoint file size/mtime inventory.

The factory owns model-specific construction. The tool does not decode native
payloads or acknowledge a checkpoint. Run with the same CPU-only, credential-free
environment as the dense audit. A restored finite sparse state is separate
from numerical reference equivalence and crash exactly-once proof.

## Successful Reader range coverage

```bash
python -m tools.validation.reconcile_reader_audit \
  --phase "$RUN_DIR/phase1_reader.json" "$PHASE1_READERS" 0 "$FIRST_CUT" \
  --phase "$RUN_DIR/phase2_reader.json" "$PHASE2_READERS" "$FIRST_CUT" "$FINAL_CUT" \
  --batch-size "$GLOBAL_BATCH_SIZE" --source-rows "$SOURCE_ROWS" \
  --require-exhaustion --output "$RUN_DIR/reader-coverage.json"
```

Each phase argument specifies an audit base path, worker count, inclusive
starting step and exclusive ending step. The tool reads the existing
`<stem>_rank_<rank><suffix>` files. Each must contain its physical `rank` and
the Reader's successful assigned-read `records`.

The validator requires contiguous phase boundaries, valid rank/step/range
metadata, unchanged source/schema identity and exactly one complete global
batch of assigned slots at each step. `--require-exhaustion` also requires
coverage from step zero through `ceil(source_rows / batch_size)`. Only the
final batch may contain slots beyond the real source row count. Output counts
distinguish real rows and padding.

This audit is written at Reader shutdown. It records successful source reads,
not Input receipt or durable model consumption. A killed process may lose the
artifact. Its JSON therefore explicitly reports
`crash_exactly_once_proven=false` and `input_receipt_proven=false`.

## Reader lane placement

```bash
python -m tools.validation.validate_reader_locality \
  "$RUN_DIR/allocation.json" --minimum-nodes 2 \
  --output "$RUN_DIR/reader-locality.json"
```

The input contains the placement fields from the public READY allocation report, compiled Reader
execution groups and explicit Reader/Input SPMD IDs. The validator checks the
plan against runtime `attested_node_ids`, requires the Input ranks to span the
requested number of nodes, and verifies every physical Reader in a compiled
lane resides on its corresponding Input rank's node. Execution groups must
partition all Reader ranks and supply multiple Readers per Input lane.

This proves placement against the actual compiled lane mapping. It does not
measure network bytes or establish physical multi-host performance. The
`tools.validation.local_ray_cluster` context manager supports a GPU simulation
on one physical host: it partitions visible GPU UUIDs and one actual CPU/memory
budget across Ray nodes, verifies GPU arithmetic and cleans up its own cluster.
It requires an explicit non-loopback host address. It never changes host
routing or shuts down unrelated clusters.

## End-to-end evidence

Applications should retain separate results for training, topology restore,
terminal validity, assigned-read coverage, placement, controlled crash/restart,
independent weight-only inference and metric comparison. Controlled-crash
acceptance needs an observed interruption beyond a known durable cut, a public
restore of that cut and reconciliation of committed source/model/optimizer
state against a reference with matching update boundaries. Replayed
speculative reads are not committed duplicate consumption.

Use an immutable source snapshot and aligned complete evaluation windows when
comparing metrics. Finite AUC/GAUC values alone do not prove improvement or
general statistical convergence.
