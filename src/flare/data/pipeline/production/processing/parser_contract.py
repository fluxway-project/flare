from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ParserContract:
    """Recipe-supplied source layout and categorical metadata contract."""

    direct_columns: Mapping[str, int]
    metadata_columns: Mapping[str, int]
    metadata_maps: Mapping[str, Mapping[str, int]]
    feature_payload_column_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.direct_columns, MappingProxyType):
            raise TypeError("direct_columns must be a MappingProxyType")
        if not isinstance(self.metadata_columns, MappingProxyType):
            raise TypeError("metadata_columns must be a MappingProxyType")
        if not isinstance(self.metadata_maps, MappingProxyType):
            raise TypeError("metadata_maps must be a MappingProxyType")
        direct = self.direct_columns
        metadata = self.metadata_columns
        maps = self.metadata_maps
        if not direct:
            raise ValueError("parser contract requires direct columns")
        if set(metadata) != set(maps):
            raise ValueError(
                "parser metadata columns and maps must have identical keys"
            )
        indexes = tuple(direct.values()) + tuple(metadata.values())
        if any(not isinstance(index, int) or index < 0 for index in indexes):
            raise ValueError(
                "parser source column indexes must be non-negative integers"
            )
        if len(indexes) != len(set(indexes)):
            raise ValueError("parser source column indexes must be unique")
        if (
            not isinstance(self.feature_payload_column_index, int)
            or self.feature_payload_column_index < 0
        ):
            raise ValueError("feature payload column index must be non-negative")
        if self.feature_payload_column_index in indexes:
            raise ValueError("feature payload column must not overlap source columns")


__all__ = ["ParserContract"]
