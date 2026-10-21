from __future__ import annotations

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from flare.jax.mesh import ParallelMesh
from flare.jax.training.engines.distributed import DistributedNnxTrainEngine
from flare.jax.training.state.layout import FlatStateLayout


def main() -> None:
    devices = tuple(jax.devices())
    if len(devices) != 2 or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("checkpoint peak probe requires exactly two visible GPUs")

    parallel_mesh = ParallelMesh.from_dp(devices=devices, axis_name="dp")
    mesh = parallel_mesh.mesh
    host = np.ones((1024, 2048), dtype=np.float32)
    layout = FlatStateLayout.from_values((host, host), mesh.size)
    engine = object.__new__(DistributedNnxTrainEngine)
    engine.parallel_mesh = parallel_mesh
    materializers = engine._build_checkpoint_value_materializers(layout)
    flat_state = tuple(
        jax.device_put(host.reshape(-1), NamedSharding(mesh, P("dp")))
        for _ in layout.leaves
    )
    jax.block_until_ready(flat_state)
    before = tuple(device.memory_stats()["bytes_in_use"] for device in devices)

    checkpoint_values = engine._global_checkpoint_values(flat_state, materializers)
    jax.block_until_ready(checkpoint_values)
    after = tuple(device.memory_stats()["bytes_in_use"] for device in devices)
    peaks = tuple(device.memory_stats()["peak_bytes_in_use"] for device in devices)

    expected_output_bytes_per_device = sum(
        host.nbytes // mesh.size for _ in layout.leaves
    )
    allocator_tolerance_bytes = 1 << 20
    for device, start, finish, peak in zip(devices, before, after, peaks, strict=True):
        limit = start + expected_output_bytes_per_device + allocator_tolerance_bytes
        if finish > limit or peak > limit:
            raise AssertionError(
                f"checkpoint capture allocated replicated payload on {device}: "
                f"before={start}, after={finish}, peak={peak}, limit={limit}"
            )
    if any(value.sharding.spec != P("dp") for value in checkpoint_values):
        raise AssertionError("checkpoint values must retain logical sharding")
    if any(value.shape != host.shape for value in checkpoint_values):
        raise AssertionError("checkpoint values must retain logical shapes")
    if any(not np.array_equal(np.asarray(value), host) for value in checkpoint_values):
        raise AssertionError("checkpoint values differ from the logical source")

    print(f"CHECKPOINT_PEAK_MEMORY=PASS before={before} after={after} peaks={peaks}")


if __name__ == "__main__":
    main()
