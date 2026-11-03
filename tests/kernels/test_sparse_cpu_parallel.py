from __future__ import annotations

import numpy as np

from flare.ops.sparse_scatter import scatter_pooled_numba
from recipes.recsys.ranking.schema.input import materialize_padded_inverse


def test_nested_inverse_materialization_preserves_padding_and_positions() -> None:
    flat = np.asarray([4, 5, 6, 7], dtype=np.int32)
    # Two rows, each with two outer children; one child has no inner values.
    nested_splits = (
        np.asarray([0, 2, 4], dtype=np.int32),
        np.asarray([0, 1, 3, 3, 4], dtype=np.int32),
    )

    result = materialize_padded_inverse(
        flat,
        nested_splits,
        (2, 2, 2),
        pad_index=9,
        batch_size=2,
        target_widths=(2, 2),
        group="test_dim_2",
    )

    np.testing.assert_array_equal(
        result,
        np.asarray([[[4, 9], [5, 6]], [[9, 9], [7, 9]]], dtype=np.int32),
    )


def test_grouped_parallel_pooled_scatter_matches_duplicate_id_sum() -> None:
    # Exercise the row-parallel path (>=4096 slots and a narrow embedding dim).
    rng = np.random.default_rng(123)
    emb_rows, batch, feature_count, max_elements, emb_dim = 17, 1024, 2, 2, 4
    pad_index = emb_rows
    inverse = rng.integers(0, emb_rows + 1, size=batch * feature_count * max_elements)
    inverse[:8] = [1, 1, 2, pad_index, 3, 1, 2, 2]
    gradient = rng.standard_normal((batch, feature_count, emb_dim), dtype=np.float32)

    actual = scatter_pooled_numba(
        emb_rows,
        emb_dim,
        inverse.astype(np.int32),
        pad_index,
        gradient,
        (feature_count, max_elements),
    )

    expected = np.zeros((emb_rows, emb_dim), dtype=np.float32)
    for slot, row in enumerate(inverse):
        if row != pad_index:
            feature = (slot // max_elements) % feature_count
            row_in_batch = slot // (feature_count * max_elements)
            expected[row] += gradient[row_in_batch, feature]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
