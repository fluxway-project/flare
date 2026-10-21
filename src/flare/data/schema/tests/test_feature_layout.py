import json
import pickle

import numpy as np
import pyarrow as pa
import pytest

from flare.data.schema import table_to_training_dict
from flare.data.schema.feature_catalog import (
    FeatureCatalog,
    FeatureDescriptor,
    FeatureOutputKind,
    FeatureRepresentation,
)
from flare.data.schema.feature_layout import build_training_feature_layout


def descriptor(code: int, name: str) -> FeatureDescriptor:
    return FeatureDescriptor(
        code=code,
        name=name,
        representation=FeatureRepresentation.FID,
        length=1,
        output_kind=FeatureOutputKind.SPARSE_I64_FIDS,
        encoder="AsItIsEncoderExpr",
        cross_type=None,
        cross_class=None,
        dependency_codes=(),
    )


def dense_descriptor(code: int, name: str, *, length: int = 1) -> FeatureDescriptor:
    return FeatureDescriptor(
        code=code,
        name=name,
        representation=FeatureRepresentation.FID,
        length=length,
        output_kind=FeatureOutputKind.DENSE_I32_INDICES,
        encoder="CategoryEncoderExpr",
        cross_type=None,
        cross_class=None,
        dependency_codes=(),
    )


def write_catalog(tmp_path, descriptors: list[FeatureDescriptor]) -> None:
    columns = ("target_label",) + tuple(item.output_column for item in descriptors)
    catalog = FeatureCatalog(
        descriptors,
        config_digest="a" * 64,
        columns=columns,
    )
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps(catalog.to_manifest()),
        encoding="utf-8",
    )


def test_feature_manifest_is_required(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="feature manifest not found"):
        FeatureCatalog.from_manifest_dir(tmp_path)


def test_layout_uses_code_derived_physical_columns(tmp_path) -> None:
    write_catalog(
        tmp_path,
        [
            dense_descriptor(55113, "feature_a"),
            descriptor(60213, "feature_a_cross"),
        ],
    )
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)

    layout = build_training_feature_layout(
        catalog,
        [("feature_a", 1)],
        {"group": ["feature_a_cross"]},
        None,
    )

    assert layout.projection == ("fid_55113", "fid_60213")
    assert layout.column_for("feature_a") == "fid_55113"
    assert layout.output_kind_for("feature_a") is FeatureOutputKind.DENSE_I32_INDICES


def test_layout_rejects_ambiguous_name_only_model_reference(tmp_path) -> None:
    write_catalog(
        tmp_path,
        [descriptor(55070, "same-name"), descriptor(55079, "same-name")],
    )
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)

    with pytest.raises(ValueError, match="ambiguous feature name"):
        build_training_feature_layout(catalog, [("same-name", 1)], None, None)


def test_layout_round_trips_through_worker_pickle(tmp_path) -> None:
    write_catalog(tmp_path, [dense_descriptor(55113, "readable-name")])
    layout = build_training_feature_layout(
        FeatureCatalog.from_manifest_dir(tmp_path),
        [("readable-name", 1)],
        None,
        None,
    )

    restored = pickle.loads(pickle.dumps(layout))

    assert restored.projection == ("fid_55113",)
    assert restored.column_for("readable-name") == "fid_55113"


def test_layout_rejects_dense_feature_in_sparse_group(tmp_path) -> None:
    write_catalog(tmp_path, [dense_descriptor(2687, "dense-category")])
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)

    with pytest.raises(
        ValueError,
        match="sparse feature 'dense-category' must use SPARSE_I64_FIDS",
    ):
        build_training_feature_layout(
            catalog,
            side_info_seq=None,
            categorical_feature_groups={"table": ["dense-category"]},
            metadata_cols=None,
        )


def test_side_info_list_padding_is_reader_contract_not_physical_metadata(
    tmp_path,
) -> None:
    write_catalog(
        tmp_path,
        [dense_descriptor(2687, "dense-category", length=2)],
    )
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)
    layout = build_training_feature_layout(
        catalog,
        side_info_seq=[("dense-category", 2)],
        categorical_feature_groups=None,
        metadata_cols=None,
    )
    table = pa.table(
        {
            "fid_2687": pa.array([[1, None]], type=pa.list_(pa.int32())),
            "target_label": pa.array([1], type=pa.int32()),
        }
    )

    batch = table_to_training_dict(
        table,
        side_info_seq=[("dense-category", 2)],
        categorical_feature_groups={},
        metadata_cols=[],
        label_column="target_label",
        feature_layout=layout,
    )
    np.testing.assert_array_equal(
        batch["side_info"], np.asarray([[2, 1]], dtype=np.int32)
    )


def test_dense_features_keep_independent_flat_positions(tmp_path) -> None:
    descriptors = [
        dense_descriptor(2687, "dense-a", length=2),
        dense_descriptor(2688, "dense-b", length=2),
    ]
    write_catalog(tmp_path, descriptors)
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)
    side_info_seq = [("dense-a", 2), ("dense-b", 2)]
    layout = build_training_feature_layout(
        catalog,
        side_info_seq=side_info_seq,
        categorical_feature_groups=None,
        metadata_cols=None,
    )
    list_metadata = {b"pad_length": b"2", b"pad_value": b"0"}
    schema = pa.schema(
        [
            pa.field("fid_2687", pa.list_(pa.int32()), metadata=list_metadata),
            pa.field("fid_2688", pa.list_(pa.int32()), metadata=list_metadata),
            pa.field("target_label", pa.int32()),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([[1, None]], type=pa.list_(pa.int32())),
            pa.array([[1, None]], type=pa.list_(pa.int32())),
            pa.array([1], type=pa.int32()),
        ],
        schema=schema,
    )

    batch = table_to_training_dict(
        table,
        side_info_seq=side_info_seq,
        categorical_feature_groups={},
        metadata_cols=[],
        label_column="target_label",
        feature_layout=layout,
    )

    np.testing.assert_array_equal(
        batch["side_info"],
        np.asarray([[2, 1, 2, 1]], dtype=np.int32),
    )


def test_side_info_accepts_iceberg_int_list_round_trip(tmp_path) -> None:
    write_catalog(tmp_path, [dense_descriptor(2687, "dense-category", length=2)])
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)
    layout = build_training_feature_layout(
        catalog,
        side_info_seq=[("dense-category", 2)],
        categorical_feature_groups=None,
        metadata_cols=None,
    )
    table = pa.table(
        {
            "fid_2687": pa.array([[7, None]], type=pa.list_(pa.int32())),
            "target_label": pa.array([1], type=pa.int32()),
        }
    )
    output = table_to_training_dict(
        table,
        side_info_seq=[("dense-category", 2)],
        categorical_feature_groups={},
        metadata_cols=[],
        label_column="target_label",
        feature_layout=layout,
    )
    np.testing.assert_array_equal(
        output["side_info"], np.asarray([[8, 1]], dtype=np.int32)
    )


def test_side_info_rejects_iceberg_long_list(tmp_path) -> None:
    write_catalog(tmp_path, [dense_descriptor(2687, "dense-category", length=2)])
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)
    layout = build_training_feature_layout(
        catalog,
        side_info_seq=[("dense-category", 2)],
        categorical_feature_groups=None,
        metadata_cols=None,
    )
    table = pa.table(
        {
            "fid_2687": pa.array([[7, None]], type=pa.list_(pa.int64())),
            "target_label": pa.array([1], type=pa.int32()),
        }
    )

    with pytest.raises(
        ValueError,
        match="side-info feature 'fid_2687' must use list<int32>",
    ):
        table_to_training_dict(
            table,
            side_info_seq=[("dense-category", 2)],
            categorical_feature_groups={},
            metadata_cols=[],
            label_column="target_label",
            feature_layout=layout,
        )


def test_side_info_rejects_negative_iceberg_int_value(tmp_path) -> None:
    write_catalog(tmp_path, [dense_descriptor(2687, "dense-category", length=1)])
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)
    layout = build_training_feature_layout(
        catalog,
        side_info_seq=[("dense-category", 1)],
        categorical_feature_groups=None,
        metadata_cols=None,
    )
    table = pa.table(
        {
            "fid_2687": pa.array([[-1]], type=pa.list_(pa.int32())),
            "target_label": pa.array([1], type=pa.int32()),
        }
    )

    with pytest.raises(ValueError, match="negative dense indices"):
        table_to_training_dict(
            table,
            side_info_seq=[("dense-category", 1)],
            categorical_feature_groups={},
            metadata_cols=[],
            label_column="target_label",
            feature_layout=layout,
        )


def test_sparse_iceberg_long_preserves_int64_bit_pattern(tmp_path) -> None:
    write_catalog(tmp_path, [descriptor(17, "sparse-id")])
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)
    layout = build_training_feature_layout(
        catalog,
        side_info_seq=None,
        categorical_feature_groups={"table": ["sparse-id"]},
        metadata_cols=None,
    )
    signed = np.asarray([-1, np.iinfo(np.int64).min, 7], dtype=np.int64)
    table = pa.table(
        {
            "fid_17": pa.array(
                [[int(value)] for value in signed], pa.list_(pa.int64())
            ),
            "target_label": pa.array([1, 1, 1], type=pa.int32()),
        }
    )

    output = table_to_training_dict(
        table,
        side_info_seq=[],
        categorical_feature_groups={"table": ["sparse-id"]},
        metadata_cols=[],
        label_column="target_label",
        feature_layout=layout,
    )

    assert output["table"].flat_values.dtype == np.int64
    np.testing.assert_array_equal(output["table"].flat_values, signed)


def test_reader_rejects_int64_system_label(tmp_path) -> None:
    write_catalog(tmp_path, [descriptor(17, "sparse-id")])
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)
    layout = build_training_feature_layout(
        catalog,
        side_info_seq=None,
        categorical_feature_groups={"table": ["sparse-id"]},
        metadata_cols=["target_label"],
        system_columns={"target_label": FeatureOutputKind.DENSE_I32_INDICES},
    )
    table = pa.table(
        {
            "fid_17": pa.array([[7]], pa.list_(pa.int64())),
            "target_label": pa.array([1], pa.int64()),
        }
    )

    with pytest.raises(ValueError, match="target_label.*int32"):
        table_to_training_dict(
            table,
            side_info_seq=[],
            categorical_feature_groups={"table": ["sparse-id"]},
            metadata_cols=["target_label"],
            label_column="target_label",
            feature_layout=layout,
        )


def test_layout_records_system_column_kind(tmp_path) -> None:
    write_catalog(tmp_path, [descriptor(17, "sparse-id")])
    layout = build_training_feature_layout(
        FeatureCatalog.from_manifest_dir(tmp_path),
        side_info_seq=None,
        categorical_feature_groups={"table": ["sparse-id"]},
        metadata_cols=["target_label"],
        system_columns={"target_label": FeatureOutputKind.DENSE_I32_INDICES},
    )

    assert layout.output_kind_for("target_label") is FeatureOutputKind.DENSE_I32_INDICES


def test_layout_rejects_sparse_feature_in_side_info(tmp_path) -> None:
    write_catalog(tmp_path, [descriptor(17, "sparse-id")])
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)

    with pytest.raises(
        ValueError,
        match="side-info feature 'sparse-id' must use DENSE_I32_INDICES",
    ):
        build_training_feature_layout(
            catalog,
            side_info_seq=[("sparse-id", 1)],
            categorical_feature_groups=None,
            metadata_cols=None,
        )


def test_layout_rejects_model_length_that_differs_from_schema(tmp_path) -> None:
    write_catalog(tmp_path, [dense_descriptor(2687, "dense-category", length=5)])
    catalog = FeatureCatalog.from_manifest_dir(tmp_path)

    with pytest.raises(
        ValueError,
        match="model=1, schema=5",
    ):
        build_training_feature_layout(
            catalog,
            side_info_seq=[("dense-category", 1)],
            categorical_feature_groups=None,
            metadata_cols=None,
        )
