from __future__ import annotations

import pyarrow as pa

from flare.data.schema.feature_catalog import FeatureOutputKind
from flare.data.schema.feature_layout import TrainingFeatureLayout
from recipes.recsys.ranking.schema.transform import make_batch_transformer_factory


def test_recipe_batch_transformer_factory_returns_callable() -> None:
    layout = TrainingFeatureLayout(
        projection=("label",),
        bindings=(("label", "label"),),
        kinds_by_name={"label": FeatureOutputKind.DENSE_I32_INDICES},
        padding_by_name={},
        null_index_sentinel=-3,
    )
    transformer = make_batch_transformer_factory(
        side_info_seq=(),
        categorical_feature_groups={},
        metadata_cols=(),
        label_column="label",
    )(layout)

    assert callable(transformer)
    output = transformer(pa.table({"label": pa.array([1], type=pa.int32())}))
    assert output["labels"].tolist() == [1]
