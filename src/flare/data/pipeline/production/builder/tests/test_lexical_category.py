import numpy as np
import pyarrow as pa
import pytest

from flare.data.pipeline.production.builder._fast_parse import (
    CategoryTable,
    MetadataTable,
    parse_arrow_direct,
)
from flare.data.pipeline.production.builder.encoding.dense import (
    CategoryEncoderExpr,
    CategoryLookupTable,
)
from flare.data.pipeline.production.builder.row_parser import FeatureBuffers, RowParser
from flare.data.pipeline.production.processing.parser_spec import (
    ParserFeatureSpec,
    ParserSpec,
)


def _parser() -> RowParser:
    spec = ParserSpec(
        features=(
            ParserFeatureSpec(code=7, name="category", length=3, is_float=False),
        ),
        feature_column_index={},
        raw_schema=pa.schema(
            [
                pa.field("category", pa.list_(pa.int64())),
                pa.field("category_index_7", pa.list_(pa.int32())),
            ]
        ),
        direct_column_index={},
        metadata_column_index={},
        feature_payload_column_index=0,
        null_sentinel=-3,
        float_null_sentinel=-3.0,
        category_mappings={7: {"10": 1, "10.0": 2, "0.5": 3, "-3": 4}},
        category_defaults={7: 0},
    )
    return RowParser(spec, {})


def test_vector_category_values_preserve_raw_lexemes() -> None:
    parser = _parser()
    rows = [
        "7" + chr(3) + "[10,10.0,0.5]",
        "7" + chr(3) + "[999]",
    ]
    result = parser.parse_to_table(pa.chunked_array([pa.array(rows)]))
    assert result.column("category").to_pylist() == [[10, 10, 0], [999]]
    assert result.column("category_index_7").to_pylist() == [[1, 2, 3], [0]]


def test_category_vector_rejects_excess_values() -> None:
    parser = _parser()
    row = "7" + chr(3) + "[10,10.0,0.5,999]"
    with pytest.raises(ValueError, match="category vector exceeds configured length"):
        parser.parse_to_table(pa.chunked_array([pa.array([row])]))


def test_trailing_tab_does_not_extend_final_category_value() -> None:
    parser = _parser()
    row = "7" + chr(3) + "[-3]" + chr(9)
    result = parser.parse_to_table(pa.chunked_array([pa.array([row])]))
    assert result.column("category").to_pylist() == [[-3]]
    assert result.column("category_index_7").to_pylist() == [[4]]


def test_vector_category_with_negative_float_sentinel() -> None:
    spec = ParserSpec(
        features=(
            ParserFeatureSpec(code=7, name="category", length=3, is_float=False),
        ),
        feature_column_index={},
        raw_schema=pa.schema(
            [
                pa.field("category", pa.list_(pa.int64())),
                pa.field("category_index_7", pa.list_(pa.int32())),
            ]
        ),
        direct_column_index={},
        metadata_column_index={},
        feature_payload_column_index=0,
        null_sentinel=-3,
        float_null_sentinel=-3.0,
        category_mappings={7: {"-3": 1, "-3.0": 2}},
        category_defaults={7: 0},
    )
    parser = RowParser(spec, {})
    rows = [
        "7" + chr(3) + "[-3.0]",
        "7" + chr(3) + "[-3]",
    ]
    result = parser.parse_to_table(pa.chunked_array([pa.array(rows)]))
    assert result.column("category").to_pylist() == [[-3], [-3]]
    assert result.column("category_index_7").to_pylist() == [[2], [1]]


def test_category_default_index_contract() -> None:
    spec = ParserSpec(
        features=(
            ParserFeatureSpec(code=7, name="category", length=1, is_float=False),
        ),
        feature_column_index={},
        raw_schema=pa.schema(
            [
                pa.field("category", pa.int64()),
                pa.field("category_index_7", pa.int32()),
            ]
        ),
        direct_column_index={},
        metadata_column_index={},
        feature_payload_column_index=0,
        null_sentinel=-3,
        float_null_sentinel=-3.0,
        category_mappings={7: {"10": 1}},
        category_defaults={7: 7},
    )
    parser = RowParser(spec, {})
    rows = [
        "7" + chr(3) + "10",
        "7" + chr(3) + "999",
    ]
    result = parser.parse_to_table(pa.chunked_array([pa.array(rows)]))
    assert result.column("category_index_7").to_pylist() == [1, 7]


def test_shared_vocabulary_preserves_feature_specific_defaults() -> None:
    spec = ParserSpec(
        features=(
            ParserFeatureSpec(code=7, name="first", length=1, is_float=False),
            ParserFeatureSpec(code=8, name="second", length=1, is_float=False),
        ),
        feature_column_index={},
        raw_schema=pa.schema(
            [
                pa.field("first", pa.int64()),
                pa.field("second", pa.int64()),
                pa.field("category_index_7", pa.int32()),
                pa.field("category_index_8", pa.int32()),
            ]
        ),
        direct_column_index={},
        metadata_column_index={},
        feature_payload_column_index=0,
        null_sentinel=-3,
        float_null_sentinel=-3.0,
        category_mappings={7: {"10": 1}, 8: {"10": 1}},
        category_defaults={7: 7, 8: 8},
    )
    parser = RowParser(spec, {})
    rows = [
        chr(2).join(("7" + chr(3) + "10", "8" + chr(3) + "10")),
        chr(2).join(("7" + chr(3) + "999", "8" + chr(3) + "999")),
    ]
    result = parser.parse_to_table(pa.chunked_array([pa.array(rows)]))
    assert result.column("category_index_7").to_pylist() == [1, 7]
    assert result.column("category_index_8").to_pylist() == [1, 8]


def test_category_defaults_must_match_vocabulary_codes() -> None:
    with pytest.raises(ValueError, match="category defaults must match"):
        CategoryTable({7: {"0": 1}}, {})


@pytest.mark.parametrize("default_index", [-1, 2**31, True])
def test_category_encoder_rejects_invalid_default_index(default_index: int) -> None:
    with pytest.raises(ValueError, match="category default must fit"):
        CategoryEncoderExpr(
            CategoryLookupTable.from_mapping({"10": 1}),
            default_index=default_index,
        )


def test_duplicate_vector_segments_cannot_exceed_feature_length() -> None:
    parser = _parser()
    row = "7" + chr(3) + "[10,10]" + chr(2) + "7" + chr(3) + "[10,10]"
    with pytest.raises(ValueError, match="category vector exceeds configured length"):
        parser.parse_to_table(pa.chunked_array([pa.array([row])]))


def test_empty_batch_and_empty_payload() -> None:
    parser = _parser()
    empty_batch = parser.parse_to_table(
        pa.chunked_array([pa.array([], type=pa.string())])
    )
    assert empty_batch.num_rows == 0
    empty_payload = parser.parse_to_table(pa.chunked_array([pa.array([""])]))
    assert empty_payload.column("category_index_7").to_pylist() == [[]]


def test_native_parser_rejects_decreasing_interior_offsets() -> None:
    with pytest.raises(ValueError, match="invalid Arrow row offsets"):
        parse_arrow_direct(
            pa.py_buffer(b"7\x0300"),
            np.array([0, 3, 2], dtype=np.int64),
            2,
            {},
            (),
            1,
            0,
            (),
            MetadataTable((), {}),
            FeatureBuffers({}),
            {},
            None,
            -3,
            -3.0,
        )


def test_native_parser_rejects_missing_and_undersized_category_outputs() -> None:
    args = (
        pa.py_buffer(b"7\x0310"),
        np.array([0, 4], dtype=np.int64),
        1,
        {7: ("category", 1, False)},
        (),
        1,
        0,
        (),
        MetadataTable((), {}),
    )
    table = CategoryTable({7: {"10": 1}}, {7: 0})
    with pytest.raises(ValueError, match="category output is missing"):
        parse_arrow_direct(*args, FeatureBuffers({}), {}, table, -3, -3.0)
    buffers = FeatureBuffers(
        {"category_index_7": np.ones(1, dtype=np.uint8)}
    )
    buffers["category_index_7"] = np.empty(0, dtype=np.int32)
    with pytest.raises(ValueError, match="category scalar buffer is too small"):
        parse_arrow_direct(*args, buffers, {}, table, -3, -3.0)


def test_metadata_table_rejects_nontext_keys_and_negative_indexes() -> None:
    with pytest.raises(TypeError, match="metadata key"):
        MetadataTable((("label", 0),), {"label": {1: 1}})
    with pytest.raises(ValueError, match="metadata index"):
        MetadataTable((("label", 0),), {"label": {"known": -1}})


def test_native_parser_requires_precompiled_metadata_table() -> None:
    with pytest.raises(TypeError):
        parse_arrow_direct(
            pa.py_buffer(b""),
            np.array([0], dtype=np.int64),
            0,
            {},
            (),
            1,
            0,
            (),
            {},
            FeatureBuffers({}),
            {},
            None,
            -3,
            -3.0,
        )


def test_numeric_only_vector_keeps_its_sentinel_filtering() -> None:
    spec = ParserSpec(
        features=(ParserFeatureSpec(code=7, name="numeric", length=3, is_float=False),),
        feature_column_index={},
        raw_schema=pa.schema([pa.field("numeric", pa.list_(pa.int64()))]),
        direct_column_index={},
        metadata_column_index={},
        feature_payload_column_index=0,
        null_sentinel=-3,
        float_null_sentinel=-3.0,
    )
    parser = RowParser(spec, {})
    row = "7" + chr(3) + "[1,-3.0,2]"
    result = parser.parse_to_table(pa.chunked_array([pa.array([row])]))
    assert result.column("numeric").to_pylist() == [[1, 2]]


@pytest.mark.parametrize(
    ("is_float", "value"),
    [
        (False, "12junk"),
        (True, "1.25junk"),
        (False, "abc"),
        (False, str(2**63)),
        (False, str(-(2**63) - 1)),
        (False, "1e1000"),
        (True, "1e1000"),
    ],
)
def test_present_invalid_numeric_token_raises(is_float: bool, value: str) -> None:
    dtype = pa.float32() if is_float else pa.int64()
    spec = ParserSpec(
        features=(ParserFeatureSpec(code=7, name="numeric", length=1, is_float=is_float),),
        feature_column_index={},
        raw_schema=pa.schema([pa.field("numeric", dtype)]),
        direct_column_index={},
        metadata_column_index={},
        feature_payload_column_index=0,
        null_sentinel=-3,
        float_null_sentinel=-3.0,
    )
    parser = RowParser(spec, {})
    with pytest.raises(ValueError, match="invalid numeric feature value"):
        parser.parse_to_table(pa.chunked_array([pa.array(["7" + chr(3) + value])]))


def test_category_token_without_required_numeric_value_raises() -> None:
    spec = ParserSpec(
        features=(ParserFeatureSpec(code=7, name="category", length=1, is_float=False),),
        feature_column_index={},
        raw_schema=pa.schema(
            [pa.field("category", pa.int64()), pa.field("category_index_7", pa.int32())]
        ),
        direct_column_index={},
        metadata_column_index={},
        feature_payload_column_index=0,
        null_sentinel=-3,
        float_null_sentinel=-3.0,
        category_mappings={7: {"abc": 1}},
        category_defaults={7: 0},
    )
    parser = RowParser(spec, {})
    with pytest.raises(ValueError, match="invalid numeric feature value"):
        parser.parse_to_table(pa.chunked_array([pa.array(["7" + chr(3) + "abc"])]))


def test_invalid_numeric_vector_token_raises() -> None:
    parser = _parser()
    with pytest.raises(ValueError, match="invalid numeric feature value"):
        parser.parse_to_table(
            pa.chunked_array([pa.array(["7" + chr(3) + "[10,abc]"])])
        )


def test_invalid_numeric_direct_value_raises() -> None:
    spec = ParserSpec(
        features=(ParserFeatureSpec(code=7, name="numeric", length=1, is_float=False),),
        feature_column_index={},
        raw_schema=pa.schema([pa.field("numeric", pa.int64()), pa.field("id", pa.int64())]),
        direct_column_index={"id": 0},
        metadata_column_index={},
        feature_payload_column_index=1,
        null_sentinel=-3,
        float_null_sentinel=-3.0,
    )
    parser = RowParser(spec, {})
    with pytest.raises(ValueError, match="invalid numeric direct value"):
        parser.parse_to_table(
            pa.chunked_array([pa.array(["abc" + chr(9) + "7" + chr(3) + "1"])])
        )


def test_signed_int64_boundaries_remain_exact() -> None:
    spec = ParserSpec(
        features=(ParserFeatureSpec(code=7, name="numeric", length=1, is_float=False),),
        feature_column_index={},
        raw_schema=pa.schema([pa.field("numeric", pa.int64())]),
        direct_column_index={},
        metadata_column_index={},
        feature_payload_column_index=0,
        null_sentinel=-3,
        float_null_sentinel=-3.0,
    )
    values = [2**63 - 1, -(2**63)]
    rows = ["7" + chr(3) + str(value) for value in values]
    result = RowParser(spec, {}).parse_to_table(pa.chunked_array([pa.array(rows)]))
    assert result.column("numeric").to_pylist() == values
