from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from types import MappingProxyType

from fluxway.common.mesh import DeviceMesh

from flare.mesh.mapping import SparsePartitionMapping


def build_1d_device_mesh(size: int, *, axis_name: str) -> DeviceMesh:
    if isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0:
        raise ValueError(f"mesh size must be a positive integer, got {size!r}")
    if not isinstance(axis_name, str) or not axis_name:
        raise ValueError("mesh axis_name must be a non-empty string")
    return DeviceMesh(device_shape=(int(size),), axis_names=(axis_name,))


def build_sparse_partition_mappings(
    worker_counts: Mapping[str, int], *, axis_name: str = "shard"
) -> Mapping[str, SparsePartitionMapping]:
    if not isinstance(worker_counts, Mapping):
        raise TypeError("worker_counts must be a mapping")
    mappings: dict[str, SparsePartitionMapping] = {}
    for group, count in worker_counts.items():
        if not isinstance(group, str) or not group:
            raise ValueError("sparse group names must be non-empty strings")
        mesh = build_1d_device_mesh(count, axis_name=axis_name)
        mappings[group] = SparsePartitionMapping.from_device_mesh(
            mesh, sparse_axes=(axis_name,)
        )
    if not mappings:
        raise ValueError("worker_counts must contain at least one sparse group")
    return MappingProxyType(mappings)


def bind_sparse_worker_meshes(
    *,
    partition_mappings: Mapping[str, SparsePartitionMapping],
    worker_counts: Mapping[str, int],
    axis_name: str = "shard",
) -> Mapping[str, DeviceMesh]:
    if not isinstance(partition_mappings, Mapping):
        raise TypeError("partition_mappings must be a mapping")
    if not isinstance(worker_counts, Mapping):
        raise TypeError("worker_counts must be a mapping")
    if set(partition_mappings) != set(worker_counts):
        raise ValueError(
            "partition mappings and sparse worker counts must cover the same groups"
        )
    meshes: dict[str, DeviceMesh] = {}
    for group, count in worker_counts.items():
        mapping = partition_mappings[group]
        if not isinstance(mapping, SparsePartitionMapping):
            raise TypeError(
                f"partition mapping for {group!r} must be SparsePartitionMapping"
            )
        mesh = build_1d_device_mesh(count, axis_name=axis_name)
        mapping.validate_device_mesh(mesh)
        meshes[group] = mesh
    return MappingProxyType(meshes)


__all__ = [
    "bind_sparse_worker_meshes",
    "build_1d_device_mesh",
    "build_sparse_partition_mappings",
]
