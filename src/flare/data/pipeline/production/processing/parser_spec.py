from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import pyarrow as pa

from flare.data.pipeline.production.builder.feature_processor import FeatureDefinition


@dataclass(frozen=True, slots=True)
class ParserFeatureSpec:
    code: int
    name: str
    length: int
    is_float: bool


@dataclass(frozen=True, slots=True)
class ParserSpec:
    features: tuple[ParserFeatureSpec, ...]
    feature_column_index: Mapping[str, int]
    raw_schema: pa.Schema
    direct_column_index: Mapping[str, int]
    metadata_column_index: Mapping[str, int]
    feature_payload_column_index: int | None

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("parser spec must contain features")
        codes = tuple(feature.code for feature in self.features)
        if tuple(sorted(codes)) != codes or len(set(codes)) != len(codes):
            raise ValueError("parser feature codes must be canonical and unique")
        all_indexes = tuple(self.direct_column_index.values()) + tuple(
            self.metadata_column_index.values()
        )
        if any(index < 0 for index in all_indexes):
            raise ValueError("parser source column indexes must be non-negative")
        if len(all_indexes) != len(set(all_indexes)):
            raise ValueError("parser source column indexes must be unique")
        feature_names = {feature.name for feature in self.features}
        direct_overlap = feature_names.intersection(self.direct_column_index)
        if direct_overlap:
            raise ValueError(
                f"direct columns overlap feature columns: {sorted(direct_overlap)!r}"
            )
        if self.feature_payload_column_index is not None and (
            self.feature_payload_column_index < 0
            or self.feature_payload_column_index in all_indexes
        ):
            raise ValueError("feature payload column must not overlap a direct column")


class ParserSpecSource(Protocol):
    @property
    def definitions(self) -> tuple[FeatureDefinition, ...]: ...

    @property
    def feature_column_index(self) -> Mapping[str, int]: ...


def build_parser_spec(
    processor: ParserSpecSource,
    *,
    metadata_columns: Mapping[str, int] | None = None,
    direct_columns: Mapping[str, int] | None = None,
    feature_payload_column_index: int | None = None,
) -> ParserSpec:
    metadata_columns = {} if metadata_columns is None else metadata_columns
    direct_columns = {} if direct_columns is None else direct_columns
    source_column_names = frozenset(direct_columns) | frozenset(metadata_columns)
    features = tuple(
        ParserFeatureSpec(
            code=definition.code,
            name=definition.name,
            length=definition.length,
            is_float=definition.dtype.startswith("float"),
        )
        for definition in processor.definitions
        if not definition.is_cross and definition.name not in source_column_names
    )
    fields: list[pa.Field] = []
    for feature in features:
        if feature.is_float:
            value_type = pa.float32()
        else:
            value_type = pa.int64()
        field_type = value_type if feature.length == 1 else pa.list_(value_type)
        fields.append(pa.field(feature.name, field_type))
    feature_names = {feature.name for feature in features}
    for name in direct_columns:
        if name not in feature_names:
            fields.append(pa.field(name, pa.int64()))
    for name in metadata_columns:
        if name in feature_names or name in direct_columns:
            raise ValueError(f"parser column {name!r} overlaps a feature column")
        fields.append(pa.field(name, pa.int32()))
    return ParserSpec(
        features=features,
        feature_column_index=MappingProxyType(
            {
                name: index
                for name, index in processor.feature_column_index.items()
                if name not in source_column_names
            }
        ),
        raw_schema=pa.schema(fields),
        direct_column_index=MappingProxyType(dict(direct_columns)),
        metadata_column_index=MappingProxyType(dict(metadata_columns)),
        feature_payload_column_index=feature_payload_column_index,
    )
