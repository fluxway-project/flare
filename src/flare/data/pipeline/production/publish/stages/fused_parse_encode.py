"""Optional in-process composition of the Parse and Encode Ray Data UDFs."""

from __future__ import annotations

import pyarrow as pa

from .feature_transform_split import EncodeUDF
from .parse import ParseUDF


class FusedParseEncodeUDF:
    """Preserve the existing UDF contracts without their Ray block handoff."""

    def __init__(
        self,
        parser_spec,
        metadata_maps: dict[str, dict[str, int]],
        encoded_plan: bytes,
        output_schema: pa.Schema,
        stage_profile_collector=None,
    ) -> None:
        self._parse = ParseUDF(
            parser_spec=parser_spec,
            metadata_maps=metadata_maps,
            stage_profile_collector=stage_profile_collector,
        )
        self._encode = EncodeUDF(
            encoded_plan=encoded_plan,
            output_schema=output_schema,
            stage_profile_collector=stage_profile_collector,
        )

    def __call__(self, batch: pa.Table) -> pa.Table:
        return self._encode(self._parse(batch))
