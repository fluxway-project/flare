from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest

from flare.data.schema.feature_catalog import FeatureOutputKind
from flare.data.schema.feature_layout import TrainingFeatureLayout
from flare.data.schema.transform import table_to_training_dict


def transform(table):
    names = tuple(name for name in table.column_names if name != "labels")
    layout = TrainingFeatureLayout(
        projection=table.column_names,
        bindings=tuple((name, name) for name in names),
        kinds_by_name={name: FeatureOutputKind.SPARSE_I64_FIDS for name in names},
    )
    return table_to_training_dict(
        table,
        (),
        {"sparse": names},
        (),
        label_column="labels",
        feature_layout=layout,
    )


def check_rows(table):
    before = table.to_pydict()
    names = tuple(name for name in table.column_names if name != "labels")
    values = []
    lengths = []
    for row in table.to_pylist():
        for name in names:
            selected = row[name] if isinstance(row[name], list) else [row[name]]
            values.extend(selected)
            lengths.append(len(selected))
    output = transform(table)
    tensor = output["sparse"]
    np.testing.assert_array_equal(
        tensor.flat_values, np.asarray(values, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        tensor.nested_splits[0], np.arange(table.num_rows + 1) * len(names)
    )
    np.testing.assert_array_equal(tensor.nested_splits[1], np.cumsum([0, *lengths]))
    assert tensor.row_count == table.num_rows
    assert tensor.feature_shape == (len(names), max(lengths, default=0))
    assert (
        tensor.flat_values.dtype == np.int64 and tensor.flat_values.flags.c_contiguous
    )
    assert all(splits.dtype == np.int32 for splits in tensor.nested_splits)
    np.testing.assert_array_equal(output["labels"], table["labels"].to_numpy())
    assert before == table.to_pydict()


def test_sparse_copy_does_not_expand_destination_indices():
    table = pa.table(
        {
            "scalar": pa.array([1, -2, 1], type=pa.int64()),
            "sequence": pa.array([[4, 4], [], [-3]], type=pa.list_(pa.int64())),
            "labels": pa.array([0, 1, 0], type=pa.int32()),
        }
    )
    with patch(
        "flare.data.schema.transform.np.repeat",
        side_effect=AssertionError("occurrence index expansion"),
    ):
        check_rows(table)


@pytest.mark.parametrize("chunked", [False, True])
def test_sparse_copy_matches_row_major_reference(chunked):
    generator = np.random.default_rng(20260917)
    for _ in range(12):
        columns = {
            "scalar": pa.array(generator.integers(-100, 100, size=32), type=pa.int64()),
            "labels": pa.array(np.arange(32, dtype=np.int32)),
        }
        for index in range(4):
            columns[f"sequence_{index}"] = pa.array(
                [
                    generator.integers(
                        -100, 100, size=generator.integers(0, 12)
                    ).tolist()
                    for _ in range(32)
                ],
                type=pa.list_(pa.int64()),
            )
        table = pa.table(columns).slice(3, 25)
        if chunked:
            table = pa.concat_tables([table.slice(0, 7), table.slice(7)])
        check_rows(table)


@pytest.mark.parametrize("rows", [0, 5])
def test_sparse_empty_lists(rows):
    check_rows(
        pa.table(
            {
                "sequence": pa.array(
                    [[] for _ in range(rows)], type=pa.list_(pa.int64())
                ),
                "labels": pa.array(np.zeros(rows, dtype=np.int32)),
            }
        )
    )


def test_sparse_null_ids_still_fail():
    table = pa.table(
        {
            "sequence": pa.array([[1, None]], type=pa.list_(pa.int64())),
            "labels": pa.array([1], type=pa.int32()),
        }
    )
    with pytest.raises(ValueError, match="null IDs"):
        transform(table)
