from __future__ import annotations

from recipes.recsys.ranking.schema.contracts import (
    RankingBatch,
    build_ranking_batch_schema,
)
from recipes.recsys.ranking.schema.input import (
    materialize_padded_inverse,
    validate_inverse_range,
)
from recipes.recsys.ranking.schema.materialization import (
    materialize_recsys_sparse_inputs,
)
from recipes.recsys.ranking.schema.sparse_features import (
    build_sparse_group_feature_widths,
    extract_sparse_base_names,
    extract_sparse_groups_and_dimensions,
    parse_sparse_group_name,
    parse_sparse_workers,
)
from recipes.recsys.ranking.schema.transform import (
    make_batch_transformer_factory,
)

__all__ = [
    "RankingBatch",
    "build_ranking_batch_schema",
    "build_sparse_group_feature_widths",
    "extract_sparse_base_names",
    "extract_sparse_groups_and_dimensions",
    "make_batch_transformer_factory",
    "materialize_padded_inverse",
    "materialize_recsys_sparse_inputs",
    "parse_sparse_group_name",
    "parse_sparse_workers",
    "validate_inverse_range",
]
