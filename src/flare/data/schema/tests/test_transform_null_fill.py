from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from flare.data.schema.feature_catalog import FeatureOutputKind
from flare.data.schema.feature_layout import TrainingFeatureLayout
from flare.data.schema.transform import _padded_list_values, table_to_training_dict


def transform(table: pa.Table) -> dict:
    layout = TrainingFeatureLayout(
        projection=("scalar", "sequence", "labels"),
        bindings=(("scalar", "scalar"), ("sequence", "sequence")),
        kinds_by_name={
            "scalar": FeatureOutputKind.DENSE_I32_INDICES,
            "sequence": FeatureOutputKind.DENSE_I32_INDICES,
        },
        padding_by_name={"scalar": -7, "sequence": -7},
        null_index_sentinel=-7,
    )
    return table_to_training_dict(
        table,
        (("scalar", 2), ("sequence", 2)),
        {},
        (),
        label_column="labels",
        feature_layout=layout,
    )


def make_table(scalars, sequences) -> pa.Table:
    return pa.table(
        {
            "scalar": pa.array(scalars, type=pa.int32()),
            "sequence": pa.array(sequences, type=pa.list_(pa.int32())),
            "labels": pa.array(list(range(len(scalars))), type=pa.int32()),
        }
    )


def test_null_free_side_info_does_not_run_fill_null() -> None:
    table = make_table([0, -7, 8], [[1, 2], [3, -7], [4, 5, 6]])
    with (
        patch("flare.data.schema.transform.pc.fill_null", wraps=pc.fill_null) as fill,
        patch(
            "flare.data.schema.transform.pc.list_slice", wraps=pc.list_slice
        ) as list_slice,
    ):
        result = transform(table)
    np.testing.assert_array_equal(
        result["side_info"], [[1, 0, 2, 3], [0, 0, 4, 0], [9, 0, 5, 6]]
    )
    assert result["side_info"].dtype == np.int32
    assert result["side_info"].flags.c_contiguous
    fill.assert_not_called()
    list_slice.assert_not_called()


@pytest.mark.parametrize("chunked", [False, True])
def test_nullable_sliced_side_info_preserves_padding(chunked: bool) -> None:
    source = make_table(
        [99, None, 3, -7, 8, 99], [[99], [2, None], [], None, [9, 10, 11], [99]]
    )
    table = source.slice(1, 4)
    if chunked:
        table = pa.concat_tables([table.slice(0, 1), table.slice(1, 3)])
    before = table.to_pydict()
    result = transform(table)
    np.testing.assert_array_equal(
        result["side_info"], [[0, 0, 3, 0], [4, 0, 0, 0], [0, 0, 0, 0], [9, 0, 10, 11]]
    )
    np.testing.assert_array_equal(result["labels"], [1, 2, 3, 4])
    assert table.to_pydict() == before


@pytest.mark.parametrize("scalars,sequences", [([-2], [[0, 1]]), ([1], [[-2, 0]])])
def test_negative_dense_indices_still_fail(scalars, sequences) -> None:
    with pytest.raises(ValueError, match="negative dense indices"):
        transform(make_table(scalars, sequences))


@pytest.mark.parametrize("width", [1, 2, 7, 20])
@pytest.mark.parametrize("padding", [-7, 0, 9])
def test_list_padding_matches_arrow_for_sliced_nullable_data(width, padding) -> None:
    generator = np.random.default_rng(20260917)
    for _ in range(12):
        rows = []
        for row_index in range(35):
            values = generator.integers(0, 100, size=generator.integers(0, 15)).tolist()
            if values and row_index % 3 == 0:
                values[0] = None
            rows.append(None if row_index % 7 == 0 else values)
        column = pa.array(rows, type=pa.list_(pa.int32())).slice(3, 28)
        pad = pa.scalar(padding, type=pa.int32())
        expected = pc.fill_null(
            pc.list_slice(column, 0, width, return_fixed_size_list=True).values, pad
        )
        actual = _padded_list_values(column, width, pad)
        assert actual.equals(expected)


def test_parent_null_hides_nonempty_children() -> None:
    column = pa.ListArray.from_arrays(
        pa.array([0, 2, 4], type=pa.int32()),
        pa.array([-2, -3, 5, 6], type=pa.int32()),
        mask=pa.array([True, False]),
    )
    pad = pa.scalar(-7, type=pa.int32())
    actual = _padded_list_values(column, 2, pad)
    assert actual.to_pylist() == [-7, -7, 5, 6]


def test_empty_list_column() -> None:
    column = pa.array([], type=pa.list_(pa.int32()))
    actual = _padded_list_values(column, 3, pa.scalar(-7, type=pa.int32()))
    assert actual.type == pa.int32() and len(actual) == 0
