from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import pyarrow as pa

from flare.data.schema import table_to_training_dict
from flare.data.schema.feature_layout import TrainingFeatureLayout


def make_batch_transformer_factory(
    *,
    side_info_seq: Sequence[tuple[str, int]],
    categorical_feature_groups: Mapping[str, Sequence[str]],
    metadata_cols: Sequence[str],
    label_column: str,
) -> Callable[[TrainingFeatureLayout], Callable[[pa.Table], Mapping[str, Any]]]:
    def create(
        feature_layout: TrainingFeatureLayout,
    ) -> Callable[[pa.Table], Mapping[str, Any]]:
        return partial(
            table_to_training_dict,
            side_info_seq=side_info_seq,
            categorical_feature_groups=categorical_feature_groups,
            metadata_cols=metadata_cols,
            label_column=label_column,
            feature_layout=feature_layout,
        )

    return create


__all__ = ["make_batch_transformer_factory"]
