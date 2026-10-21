import json

import pytest

from flare.data.schema.feature_catalog import (
    FeatureCatalog,
    FeatureDescriptor,
    FeatureOutputKind,
    FeatureRepresentation,
    storage_column_name,
)


def descriptor(
    code: int,
    name: str,
    representation: FeatureRepresentation = FeatureRepresentation.FID,
) -> FeatureDescriptor:
    is_dense = representation is FeatureRepresentation.DENSE
    return FeatureDescriptor(
        code=code,
        name=name,
        representation=representation,
        length=1,
        output_kind=(
            FeatureOutputKind.DENSE_FLOAT
            if is_dense
            else FeatureOutputKind.SPARSE_I64_FIDS
        ),
        encoder="RawFloatEncoderExpr" if is_dense else "HashEncoderExpr",
        cross_type=None,
        cross_class=None,
        dependency_codes=(),
    )


def test_storage_column_name_uses_only_identity_and_representation() -> None:
    assert storage_column_name(55113, FeatureRepresentation.FID) == "fid_55113"
    assert storage_column_name(59005, FeatureRepresentation.DENSE) == "dense_59005"


def test_catalog_allows_same_display_name_for_different_codes() -> None:
    catalog = FeatureCatalog(
        [descriptor(55070, "same-name"), descriptor(55079, "same-name")],
        config_digest="a" * 64,
        columns=("fid_55070", "fid_55079"),
    )
    assert catalog.require(55070).name == "same-name"
    assert catalog.require(55079).name == "same-name"
    with pytest.raises(ValueError, match="ambiguous feature name"):
        catalog.resolve_unique_name("same-name")


def test_catalog_rejects_duplicate_code() -> None:
    with pytest.raises(ValueError, match="duplicate feature code 55070"):
        FeatureCatalog(
            [descriptor(55070, "first"), descriptor(55070, "second")],
            config_digest="a" * 64,
            columns=("fid_55070",),
        )


def test_manifest_round_trip_preserves_human_meaning(tmp_path) -> None:
    source = FeatureCatalog(
        [
            descriptor(55113, "con_city-expect_city"),
            descriptor(
                59005,
                "exp_paddfchat_times_3d/exp_paddf_times_3d",
                FeatureRepresentation.DENSE,
            ),
        ],
        config_digest="b" * 64,
        columns=("label_primary", "detailed_label_primary", "fid_55113", "dense_59005"),
    )
    source.write_manifest(tmp_path)

    restored = FeatureCatalog.from_manifest_dir(tmp_path)

    assert restored.descriptors == source.descriptors
    assert restored.columns == source.columns
    assert restored.config_digest == "b" * 64
    assert restored.require(55113).name == "con_city-expect_city"
    assert restored.require(55113).output_column == "fid_55113"
    assert restored.require(59005).output_column == "dense_59005"


@pytest.mark.parametrize("field", ["schema_version", "unexpected"])
def test_manifest_rejects_extra_root_fields(tmp_path, field) -> None:
    payload = FeatureCatalog(
        [descriptor(55113, "con_city-expect_city")],
        config_digest="c" * 64,
        columns=("fid_55113",),
    ).to_manifest()
    payload[field] = 1
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature manifest fields do not match"):
        FeatureCatalog.from_manifest_dir(tmp_path)


@pytest.mark.parametrize("field", ["cross_type", "cross_class", "dependency_codes"])
def test_manifest_rejects_missing_entry_fields(tmp_path, field) -> None:
    payload = FeatureCatalog(
        [descriptor(55113, "readable-name")],
        config_digest="c" * 64,
        columns=("fid_55113",),
    ).to_manifest()
    del payload["features"]["55113"][field]
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="entry fields do not match"):
        FeatureCatalog.from_manifest_dir(tmp_path)


def test_manifest_rejects_extra_entry_fields(tmp_path) -> None:
    payload = FeatureCatalog(
        [descriptor(55113, "readable-name")],
        config_digest="c" * 64,
        columns=("fid_55113",),
    ).to_manifest()
    payload["features"]["55113"]["unexpected"] = 1
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="entry fields do not match"):
        FeatureCatalog.from_manifest_dir(tmp_path)


def test_manifest_rejects_column_not_derived_from_code(tmp_path) -> None:
    payload = FeatureCatalog(
        [descriptor(55113, "con_city-expect_city")],
        config_digest="c" * 64,
        columns=("fid_55113",),
    ).to_manifest()
    payload["features"]["55113"]["output_column"] = "con_city_x_expect_city_fid"
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_column"):
        FeatureCatalog.from_manifest_dir(tmp_path)
