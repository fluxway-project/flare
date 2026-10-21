# Mesh and inference weights

Fluxway's `DeviceMesh` owns graph worker coordinates, partition routing, and
placement. `flare.mesh` contains thin constructors and sparse partition
projections; it does not implement a scheduler or distributed runtime.
`flare.jax.mesh` bridges a Fluxway mesh to actual JAX devices and provides the
physical `ParallelMesh` used by JAX computation.

## Graph topology

Dense data parallelism and sparse table ownership are separate domains. A
dense `dp` mesh must not validate a sparse table's `shard` mapping. Build the
mapping from the requested sparse topology and validate it against the actual
sparse worker mesh when composing the graph:

```python
from flare.mesh import (
    bind_sparse_worker_meshes,
    build_1d_device_mesh,
    build_sparse_partition_mappings,
)

dense_mesh = build_1d_device_mesh(1, axis_name="dp")
sparse_workers = {"users": 2, "items": 1}
sparse_mappings = build_sparse_partition_mappings(sparse_workers)
sparse_meshes = bind_sparse_worker_meshes(
    partition_mappings=sparse_mappings,
    worker_counts=sparse_workers,
)
```

Use each returned mesh with its corresponding Fluxway SPMD. Sparse worker
counts may differ from dense DP. Invalid sizes, missing groups, and mapping
shape/axis mismatches raise errors before worker startup.

## JAX devices

Local independent inference uses a one-device compute mesh inside each worker:

```python
import jax
from flare.jax.mesh import ParallelMesh

local_mesh = ParallelMesh.from_dp(
    size=1,
    devices=(jax.local_devices()[0],),
    axis_name="dp",
).mesh
```

The graph's global `dp` mesh continues to own transport. Local forwards do not
perform cross-worker JAX collectives. Distributed training instead initializes
its JAX process topology and supplies the corresponding physical devices.
`materialize_jax_mesh` requires an explicit device mapping for multiple or
non-local ranks; it does not guess which physical devices logical IDs mean.
Single-device replicated execution uses ZeRO stage 0. ZeRO stages 1/2/3 require
an explicit distributed execution binding.

## Independent weight loading

Inference receives a model definition, immutable weight asset path, and its own
topology. A published asset has exactly `model/` and `sparse/` subdirectories:
the model adapter restores model-only Orbax state, and sparse adapters restore
Lance inference weights with owner-provided partition options. Inference does
not read training checkpoint refs, optimizer state, source positions, or
controller metadata.

```python
from sparse_embedding.checkpoint import PartitionSpec
from flare.inference.weights import bind_inference_weights
from flare.jax.layers import NnxModelDefinition

model_definition = NnxModelDefinition.from_factory(
    model_builder,
    sparse_groups=tuple(sparse_workers),
)
weights = bind_inference_weights(
    weight_path,
    model_definition,
    dense_mesh,
    partition_mappings=sparse_mappings,
    sparse_partition_assignment_spec=PartitionSpec.modulo(),
    key_dtype="int64",
)
model_state = weights.model.load(model_definition, local_mesh)
model = model_definition.materialize(model_state)
```

`weight_path` identifies an immutable `steps/<step>` directory. Group names in
the mappings must match the model's sparse groups. Compose the sparse workers
with their validated meshes and the bound sparse restore options; the storage
adapter owns loading and partition filtering. Invalid schemas, changed asset
contents, missing weights, and incompatible mappings fail immediately.
