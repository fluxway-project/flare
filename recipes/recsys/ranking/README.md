# Shared recommendation graph components

Applications supply their model, objective, feature bindings and source.
`RankingTask` binds an application label column and grouping column to a
`RankingObjective`. The same task feeds training, evaluation and graph export.
See the [shared recipe guide](../README.md) for dependencies and a synthetic example.

## Application Boundary

`graph.pipeline` owns Reader, Input and sparse-table composition. Call
`build_pipeline_nodes` with the source factory, resource configuration and
native sparse assignment contract, then pass `ranking_task` to
`build_pipeline_inputs`. The task creates the batch transformer, projection
and group binding together. Workers receive generic `labels` and `group_id`
ports. Configuration file formats and physical field names belong to the
application adapter.

`MultiTargetCrossEntropyLoss.create` receives explicit `target_names` in model
output order, a label map and target weights. It validates complete target
coverage and builds the single label matrix consumed by both loss and metrics.
Dictionary order in a configuration file does not determine output order.
Applications can supply another callable implementing `RankingObjective`.

Sample weights must be finite. Positive weights define positive labels and
negative weights define negative labels. An explicit zero weight masks that
label's loss and logits gradient for its target. Training and evaluation use
the same binary label matrix, defined by `weight > 0`.

`BaseRankingRecipe` requires `model_builder()` and `ranking_task()`; it never
loads application configuration. Its TaskSpec and synthetic batches expose
`labels` and `row_weights` independently of physical column names.

Placement exclusions are application inputs. Shared assembly defaults to an
empty exclusion set and never selects tables by a business-specific name.

## Build the graph and placement

Add the source, Input, embedding, compute and metric operators before building
placement. `assemble_ranking_training_graph` adds the training/metric operators,
builds the TaskGraph once and returns `task_graph`, `placement`, `training`
and `metrics`. Start that returned graph with that placement recipe.

For a graph assembled directly, pass its compiled graph to the shared owner:

```python
from flare.runtime.fluxway.placement import build_affinity_placement

task_graph = graph.build()
placement = build_affinity_placement(
    recipe_id="training",
    compiled_graph=task_graph.require_compiled_graph(),
    reader_spmd=reader,
    input_spmd=input_node,
    sparse_table_spmds=tables,
    compute_spmd=compute,
    policy="feasible",
)
```

The handles above identify nodes already added to `graph`; `tables` maps
feature-group names to their embedding nodes. The Reader must expose its
compiled `read` operator. Its ExecutionLayout supplies execution groups and
logical lanes. Each lane's Reader ranks are constrained to the same node as
the corresponding Input rank using Fluxway's native placement expressions.
The logical lane count must equal the Input worker count. A mismatch raises
before worker startup. Placement does not reconstruct execution groups from
worker-count arithmetic or assume contiguous physical ranks.

Both `balanced` and `feasible` policies enforce this invariant and the existing
role support constraints. Sparse/compute transport follows its own partition
dependencies; Reader locality does not imply that all graph traffic is local.
The constraints do not change resources, objectives, async transport or
checkpoint semantics. Feasibility still requires enough real cluster capacity.

## Feature and storage schemas

`schema.sparse_features` owns feature-group interpretation:

| API | Contract |
| --- | --- |
| `parse_sparse_group_name` | Parses `{base_name}_dim_{dimension}` and rejects invalid names or nonpositive dimensions. |
| `extract_sparse_groups_and_dimensions` | Preserves model feature-group order and produces sorted, distinct native storage dimensions for each base table. |
| `parse_sparse_workers` | Validates sparse worker counts and explicit group overrides. |

Model feature order and sorted storage dimensions are separate contracts.
Do not reorder model features to satisfy the native storage schema. Assigned
Reader payloads preserve the global valid-prefix count in one int32
`samples_valid_count` scalar. Losses, sparse preparation and metrics mask
padding with this count, including padding-only lanes in the terminal batch.

## Evaluation and metrics

Each `RecsysBackboneEvalSPMD` constructs one `nnx.jit` forward and sigmoid
projection. Model state and input arrays are explicit arguments. Compatible
graphs, shapes and dtypes reuse compiled execution with current values;
changed signatures can retrace. The first call includes compilation. Scores
are materialized on the host before releasing input leases, so that output
boundary waits for device completion.

Train and Eval share `RayDataMetricsComputer`. Dataset construction happens
inside its resource-limit scope. It coalesces prediction batches once per
metric window and reuses the materialized blocks across exact per-target AUC
sorts and GAUC grouping. The configured group and weighting semantics remain
unchanged. A full-source single window still collects every prediction before
starting metric reduction. Block reuse and compiled forwards do not establish
overlap between that final reduction and inference.

Inference consumes a model definition and immutable `steps/<step>` weight
asset with `model/` and `sparse/` contents. It supplies its own source and
topology and does not read training refs, optimizer state or source cursors.
Generic weight and mesh contracts are documented in
[mesh and inference](../../../docs/mesh_and_inference.md).
