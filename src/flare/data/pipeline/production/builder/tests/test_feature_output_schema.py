from flare.data.pipeline.production.builder.encoding.base import SchemaOnlyEncoderExpr
from flare.data.pipeline.production.builder.feature_processor import (
    FeatureDefinition,
    FeatureProcessor,
)
from flare.data.schema.feature_catalog import FeatureOutputKind


def test_feature_output_schema_uses_code_for_same_name_features() -> None:
    processor = object.__new__(FeatureProcessor)
    processor._definitions = {
        1: FeatureDefinition(1, "a-b", 1, "int64"),
        2: FeatureDefinition(2, "a_x_b", 1, "int64"),
    }
    processor._metadata_columns = frozenset()
    processor._code_to_encoder = {
        1: SchemaOnlyEncoderExpr(FeatureOutputKind.SPARSE_I64_FIDS),
        2: SchemaOnlyEncoderExpr(FeatureOutputKind.SPARSE_I64_FIDS),
    }

    schema = processor.build_vortex_schema()

    assert schema.names == ["fid_1", "fid_2"]
