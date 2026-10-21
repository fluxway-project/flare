from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from sparse_embedding.checkpoint import (
    PartitionSpec,
    RestorePlan,
)

from flare.mesh import SparsePartitionMapping
from flare.runtime.sparse_precision import require_key_dtype


def build_sparse_restore_plans(
    *,
    partition_mappings: Mapping[str, SparsePartitionMapping],
    assignment_spec: PartitionSpec,
    key_dtype: str,
) -> Mapping[str, Mapping[int, bytes]]:
    if not isinstance(partition_mappings, Mapping) or not partition_mappings:
        raise ValueError("partition_mappings must contain at least one sparse group")
    if not isinstance(assignment_spec, PartitionSpec):
        raise TypeError("assignment_spec must be a PartitionSpec")
    key_dtype = require_key_dtype(key_dtype)
    restore_plans_by_group: dict[str, Mapping[int, bytes]] = {}
    for group, mapping in sorted(partition_mappings.items()):
        if not isinstance(group, str) or not group:
            raise ValueError("sparse group names must be non-empty strings")
        if not isinstance(mapping, SparsePartitionMapping):
            raise TypeError(
                f"partition mapping for {group!r} must be SparsePartitionMapping"
            )
        partition_count = mapping.partition_count
        restore_plans = {
            logical_partition_id: RestorePlan.from_partition(
                spec=assignment_spec,
                partition_count=partition_count,
                partition_id=logical_partition_id,
                key_dtype=key_dtype,
            ).encode()
            for logical_partition_id in range(partition_count)
        }
        restore_plans_by_group[group] = MappingProxyType(restore_plans)
    return MappingProxyType(restore_plans_by_group)


__all__ = ["build_sparse_restore_plans"]
