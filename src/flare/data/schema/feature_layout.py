from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

from flare.data.pipeline.production.builder.constants import NULL_FID_SENTINEL
from flare.data.schema.feature_catalog import (
    FeatureCatalog,
    FeatureOutputKind,
    validate_storage_columns,
)
from flare.data.schema.virtual_pad import ArrowVirtualPadBatchFactory


class TrainingFeatureLayout:
    __slots__ = (
        "_bindings",
        "_columns_by_name",
        "_kinds_by_name",
        "_null_index_sentinel",
        "_padding_by_name",
        "_projection",
    )

    def __init__(
        self,
        *,
        projection: Sequence[str],
        bindings: Sequence[tuple[str, str]],
        kinds_by_name: Mapping[str, FeatureOutputKind],
        padding_by_name: Mapping[str, int] | None = None,
        null_index_sentinel: int = NULL_FID_SENTINEL,
    ) -> None:
        self._projection = validate_storage_columns(projection)
        columns_by_name: dict[str, str] = {}
        for logical_name, physical_column in bindings:
            if logical_name in columns_by_name:
                if columns_by_name[logical_name] != physical_column:
                    raise ValueError(
                        f"feature {logical_name!r} has conflicting columns"
                    )
                continue
            columns_by_name[logical_name] = physical_column
        self._bindings = tuple(columns_by_name.items())
        self._columns_by_name = columns_by_name
        self._kinds_by_name = dict(kinds_by_name)
        self._padding_by_name = dict(padding_by_name or {})
        if (
            isinstance(null_index_sentinel, bool)
            or not isinstance(null_index_sentinel, int)
            or null_index_sentinel >= 0
        ):
            raise ValueError(
                "null_index_sentinel must be a negative integer, "
                f"got {null_index_sentinel!r}"
            )
        self._null_index_sentinel = int(null_index_sentinel)

    @property
    def projection(self) -> tuple[str, ...]:
        return self._projection

    @property
    def bindings(self) -> tuple[tuple[str, str], ...]:
        return self._bindings

    def column_for(self, logical_name: str) -> str:
        try:
            return self._columns_by_name[logical_name]
        except KeyError as exc:
            raise ValueError(
                f"feature {logical_name!r} is not part of the training layout"
            ) from exc

    def padding_for(self, logical_name: str) -> int:
        try:
            return self._padding_by_name[logical_name]
        except KeyError as exc:
            raise ValueError(
                f"feature {logical_name!r} has no Dense padding contract"
            ) from exc

    def output_kind_for(self, logical_name: str) -> FeatureOutputKind:
        try:
            return self._kinds_by_name[logical_name]
        except KeyError as exc:
            raise ValueError(
                f"feature {logical_name!r} has no logical output kind"
            ) from exc

    @property
    def null_index_sentinel(self) -> int:
        return self._null_index_sentinel

    def create_virtual_pad_batch(
        self, schema: pa.Schema, projection: Sequence[str], count: int
    ) -> pa.RecordBatch:
        if tuple(projection) != self.projection:
            raise ValueError(
                "virtual pad projection differs from training feature layout"
            )
        scalar_types = {
            FeatureOutputKind.SPARSE_I64_FIDS: pa.int64(),
            FeatureOutputKind.DENSE_I32_INDICES: pa.int32(),
            FeatureOutputKind.DENSE_FLOAT: pa.float32(),
        }
        pad_values = {}
        for name, column in self.bindings:
            kind = self.output_kind_for(name)
            scalar_type = scalar_types[kind]
            field_type = schema.field(column).type
            layers = []
            while (
                pa.types.is_list(field_type)
                or pa.types.is_large_list(field_type)
                or pa.types.is_fixed_size_list(field_type)
            ):
                layers.append(
                    field_type.list_size
                    if pa.types.is_fixed_size_list(field_type)
                    else 1
                )
                field_type = field_type.value_type
            if field_type != scalar_type:
                raise ValueError(
                    f"virtual pad feature {name!r} requires {scalar_type}, got {field_type}"
                )
            value = (
                float(self.null_index_sentinel)
                if kind is FeatureOutputKind.DENSE_FLOAT
                else self.null_index_sentinel
            )
            for width in reversed(layers):
                value = [value] * width
            pad_values[column] = value
        return ArrowVirtualPadBatchFactory.create_batch(
            schema, projection, count, pad_values=pad_values, validity_column=None
        )


def build_training_feature_layout(
    catalog: FeatureCatalog,
    side_info_seq: Sequence[tuple[str, int]] | None = None,
    categorical_feature_groups: Mapping[str, Sequence[str]] | None = None,
    metadata_cols: Sequence[str] | None = None,
    extra_required_names: Sequence[str] | None = None,
    system_columns: Mapping[str, FeatureOutputKind] | None = None,
    null_index_sentinel: int = NULL_FID_SENTINEL,
) -> TrainingFeatureLayout:
    if side_info_seq:
        for name, length in side_info_seq:
            descriptor = catalog.require(catalog.resolve_unique_name(name))
            if descriptor.output_kind is not FeatureOutputKind.DENSE_I32_INDICES:
                raise ValueError(
                    f"side-info feature {name!r} must use DENSE_I32_INDICES, "
                    f"got {descriptor.output_kind.value}"
                )
            if descriptor.length != length:
                raise ValueError(
                    f"side-info feature {name!r} length differs from schema: "
                    f"model={length}, schema={descriptor.length}"
                )
    if categorical_feature_groups:
        for features in categorical_feature_groups.values():
            for name in features:
                descriptor = catalog.require(catalog.resolve_unique_name(name))
                if descriptor.output_kind is not FeatureOutputKind.SPARSE_I64_FIDS:
                    raise ValueError(
                        f"sparse feature {name!r} must use SPARSE_I64_FIDS, "
                        f"got {descriptor.output_kind.value}"
                    )

    requested_names: list[str] = []
    if side_info_seq:
        requested_names.extend(name for name, _length in side_info_seq)
    if categorical_feature_groups:
        for features in categorical_feature_groups.values():
            requested_names.extend(features)
    if metadata_cols:
        requested_names.extend(metadata_cols)
    if extra_required_names:
        requested_names.extend(extra_required_names)

    bindings: list[tuple[str, str]] = []
    projection: list[str] = []
    for name in dict.fromkeys(requested_names):
        column = catalog.column_for_unique_name(name, system_columns=system_columns)
        bindings.append((name, column))
        if column not in projection:
            projection.append(column)
    padding_by_name = {
        name: 0
        for name, _length in (side_info_seq or ())
        if catalog.require(catalog.resolve_unique_name(name)).output_kind
        is FeatureOutputKind.DENSE_I32_INDICES
    }
    kinds_by_name = {
        name: catalog.output_kind_for_unique_name(name, system_columns=system_columns)
        for name in dict.fromkeys(requested_names)
    }
    return TrainingFeatureLayout(
        projection=projection,
        bindings=bindings,
        kinds_by_name=kinds_by_name,
        padding_by_name=padding_by_name,
        null_index_sentinel=null_index_sentinel,
    )
