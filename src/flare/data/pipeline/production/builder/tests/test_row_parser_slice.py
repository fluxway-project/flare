from __future__ import annotations

import pyarrow as pa
import pytest

from flare.data.pipeline.production.builder.row_parser import RowParser
from flare.data.pipeline.production.processing.parser_spec import (
    ParserFeatureSpec,
    ParserSpec,
)


@pytest.mark.parametrize("arrow_type", [pa.string(), pa.large_string()])
@pytest.mark.parametrize("chunked", [False, True])
def test_arrow_slice_preserves_selected_rows(arrow_type, chunked):
    spec = ParserSpec(
        features=(ParserFeatureSpec(101, "feature", 1, False),),
        feature_column_index={"feature": 0},
        raw_schema=pa.schema(
            [
                ("feature", pa.int64()),
                ("id", pa.int64()),
                ("label", pa.int32()),
            ]
        ),
        direct_column_index={"id": 0},
        metadata_column_index={"label": 1},
        feature_payload_column_index=2,
    )
    parser = RowParser(spec, {"label": {"a": 0, "b": 1}})
    lines = pa.array(
        [
            "11\ta\t101\x031",
            "22\tb\t101\x032",
            "33\ta\t101\x033",
        ],
        type=arrow_type,
    ).slice(1, 1)
    if chunked:
        lines = pa.chunked_array([lines])
    features, labels, count = parser.parse_arrow(lines)
    assert count == 1
    assert features["id"].tolist() == [22]
    assert features["feature"].tolist() == [2]
    assert labels["label"].tolist() == [1]
