import json

import pytest

from flare.data.schema.feature_catalog import (
    FeatureCatalog,
    FeatureDescriptor,
    FeatureOutputKind,
    FeatureRepresentation,
)
from tools.validation import validate_iceberg


def test_schema_validation_rejects_invalid_feature_mapping_before_catalog_access(
    tmp_path, monkeypatch
) -> None:
    catalog = FeatureCatalog(
        (
            FeatureDescriptor(
                code=1,
                name="feature",
                representation=FeatureRepresentation.FID,
                length=1,
                output_kind=FeatureOutputKind.SPARSE_I64_FIDS,
                encoder="HashEncoderExpr",
                cross_type=None,
                cross_class=None,
                dependency_codes=(),
            ),
        ),
        config_digest="a" * 64,
        columns=("fid_1",),
    )
    manifest_path = catalog.write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["features"]["1"]["output_column"] = "unrelated_column"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    def reject_catalog_access(*args, **kwargs):
        raise AssertionError("invalid feature mapping reached catalog access")

    monkeypatch.setattr(validate_iceberg, "load_iceberg_table", reject_catalog_access)

    with pytest.raises(ValueError, match="output_column must be"):
        validate_iceberg.validate_schema(
            table_identifier="test.events",
            catalog_uri="http://unused.invalid",
            manifest_dir=tmp_path,
        )
