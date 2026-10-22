from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from flare.training.contracts.ports import TensorSpec
from flare.training.contracts.schema import BatchFieldSpec, BatchSchema


@dataclass(frozen=True)
class RankingBatch:
    sparse_features: Mapping[str, Any]
    side_info_features: Mapping[str, Any]
    labels: Mapping[str, Any]
    sample_weights: Mapping[str, Any] | None = None
    group_ids: Any | None = None
    valid_count: int | None = None


def build_ranking_batch_schema(
    sparse_groups: Sequence[str],
    *,
    side_info_groups: Sequence[str] = ("side_info",),
    include_group_id: bool = True,
) -> BatchSchema:
    fields = {
        group: BatchFieldSpec(
            name=group, spec=TensorSpec(shape=(None,), dtype="int64"), is_sparse=True
        )
        for group in sparse_groups
    }
    for side_group in side_info_groups:
        fields[side_group] = BatchFieldSpec(
            name=side_group, spec=TensorSpec(shape=(None, None), dtype="float32")
        )
    fields["labels"] = BatchFieldSpec(
        name="labels", spec=TensorSpec(shape=(None,), dtype="int32"), is_label=True
    )
    if include_group_id:
        fields["group_id"] = BatchFieldSpec(
            name="group_id", spec=TensorSpec(shape=(None,), dtype="int64")
        )
    fields["samples_valid_count"] = BatchFieldSpec(
        name="samples_valid_count", spec=TensorSpec(shape=(), dtype="int32")
    )
    return BatchSchema(fields=fields)


__all__ = ["RankingBatch", "build_ranking_batch_schema"]
