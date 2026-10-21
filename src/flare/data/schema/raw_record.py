from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import pyarrow as pa


@dataclass(frozen=True, slots=True)
class ParserFeatureSpec:
    code: int
    name: str
    length: int
    is_float: bool

    def __post_init__(self) -> None:
        if self.code <= 0:
            raise ValueError(f"feature code must be positive: {self.code}")
        if not self.name:
            raise ValueError("feature name must not be empty")
        if self.length <= 0:
            raise ValueError(f"feature length must be positive: {self.length}")


@dataclass(frozen=True, slots=True)
class ParserSpec:
    features: tuple[ParserFeatureSpec, ...]
    feature_column_index: Mapping[str, int]
    raw_schema: pa.Schema

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("parser spec must contain features")
        codes = tuple(feature.code for feature in self.features)
        if tuple(sorted(codes)) != codes or len(set(codes)) != len(codes):
            raise ValueError("parser feature codes must be canonical and unique")
        if not isinstance(self.raw_schema, pa.Schema):
            raise TypeError("raw_schema must be a PyArrow Schema")
        if not isinstance(self.feature_column_index, MappingProxyType):
            raise TypeError("feature_column_index must be a MappingProxyType")


@dataclass(frozen=True, slots=True)
class RawRecordSchema:
    """Recipe-neutral schema for raw records before feature processing.

    A Recipe owns the concrete field names and Arrow types by constructing this
    contract.  The production parser compiler may derive a processing plan from
    it, but this schema does not depend on a business Recipe or on a loader.
    """

    field_specs: tuple[ParserFeatureSpec, ...]
    column_index: Mapping[str, int]
    arrow_schema: pa.Schema

    def __post_init__(self) -> None:
        if not self.field_specs:
            raise ValueError("raw record schema must contain fields")
        if not isinstance(self.arrow_schema, pa.Schema):
            raise TypeError("arrow_schema must be a PyArrow Schema")
        names = tuple(field.name for field in self.arrow_schema)
        if len(set(names)) != len(names):
            raise ValueError("raw record schema fields must be unique")
        indexes = tuple(self.column_index.values())
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in indexes
        ):
            raise ValueError("raw record column indexes must be non-negative integers")
        if len(indexes) != len(set(indexes)):
            raise ValueError("raw record column indexes must be unique")
        if set(self.column_index) != {field.name for field in self.field_specs}:
            raise ValueError("raw record column index must cover every field spec")
        if not isinstance(self.column_index, MappingProxyType):
            raise TypeError("column_index must be a MappingProxyType")

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.field_specs)

    def validate_columns(self, columns: Sequence[str]) -> None:
        actual = tuple(columns)
        expected = tuple(self.arrow_schema.names)
        if actual != expected:
            raise ValueError(
                f"raw record columns mismatch: expected={expected!r} actual={actual!r}"
            )
