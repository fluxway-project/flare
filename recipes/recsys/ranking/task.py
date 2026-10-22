from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pyarrow as pa

from flare.data.schema.feature_layout import TrainingFeatureLayout
from recipes.recsys.ranking.schema.transform import make_batch_transformer_factory


class RankingObjective(Protocol):
    @property
    def targets(self) -> tuple[str, ...]: ...

    @property
    def target_labels(self) -> np.ndarray: ...

    def __call__(self, logits: Any, labels: Any, row_weights: Any, /) -> Any: ...


@dataclass(frozen=True, slots=True)
class RankingTask:
    label_column: str
    group_id_column: str
    objective: RankingObjective

    def __post_init__(self) -> None:
        for name in ("label_column", "group_id_column"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.label_column == self.group_id_column:
            raise ValueError("label and group ID columns must be distinct")
        if not callable(self.objective):
            raise TypeError("ranking objective must be callable")
        labels = np.asarray(self.objective.target_labels)
        targets = self.objective.targets
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("ranking objective must define distinct targets")
        if labels.ndim != 2 or labels.shape[0] != len(targets) or not labels.shape[1]:
            raise ValueError(
                "target_labels must have one row per target and label columns"
            )
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError("target_labels must be binary")

    def batch_transformer_factory(
        self,
        *,
        side_info_seq: Sequence[tuple[str, int]],
        categorical_feature_groups: Mapping[str, Sequence[str]],
    ) -> Callable[[TrainingFeatureLayout], Callable[[pa.Table], Mapping[str, Any]]]:
        return make_batch_transformer_factory(
            side_info_seq=side_info_seq,
            categorical_feature_groups=categorical_feature_groups,
            metadata_cols=(self.group_id_column,),
            label_column=self.label_column,
        )


__all__ = ["RankingObjective", "RankingTask"]
