"""Framework-neutral physical-column decoding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa


@dataclass(frozen=True, slots=True)
class FieldLayout:
    physical_name: str
    dtype: str
    shape: Literal["scalar", "fixed_list", "ragged"] = "scalar"
    width: int | None = None
    padding: int | float | None = None

    def __post_init__(self) -> None:
        if not self.physical_name or self.physical_name.strip() != self.physical_name:
            raise ValueError("layout physical_name must be canonical")
        if self.shape not in {"scalar", "fixed_list", "ragged"}:
            raise ValueError(f"unsupported layout shape: {self.shape!r}")
        if self.shape == "fixed_list" and (self.width is None or self.width <= 0):
            raise ValueError("fixed_list layout requires a positive width")
        if self.shape != "fixed_list" and self.width is not None:
            raise ValueError("only fixed_list layout may declare width")


@dataclass(frozen=True, slots=True)
class DecodedField:
    physical_name: str
    values: np.ndarray
    offsets: np.ndarray | None = None
    validity: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class DecodedBatch:
    row_count: int
    fields: tuple[DecodedField, ...]

    def __post_init__(self) -> None:
        if self.row_count <= 0:
            raise ValueError("decoded batch row_count must be positive")
        if not self.fields:
            raise ValueError("decoded batch must contain fields")

    def field(self, physical_name: str) -> DecodedField:
        for field in self.fields:
            if field.physical_name == physical_name:
                return field
        raise KeyError(physical_name)


@dataclass(frozen=True, slots=True)
class BatchLayoutSpec:
    fields: tuple[FieldLayout, ...]
    projection: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fields or not self.projection:
            raise ValueError("layout fields and projection must not be empty")
        names = tuple(field.physical_name for field in self.fields)
        if names != self.projection:
            raise ValueError("layout projection must exactly match field order")
        if len(set(names)) != len(names):
            raise ValueError("layout physical names must be unique")

    def decode(self, batch: pa.RecordBatch) -> DecodedBatch:
        if not isinstance(batch, pa.RecordBatch):
            raise TypeError("BatchLayoutSpec.decode expects pyarrow.RecordBatch")
        if tuple(batch.schema.names) != self.projection:
            raise ValueError(
                f"physical projection mismatch: expected={self.projection!r} "
                f"actual={tuple(batch.schema.names)!r}"
            )
        decoded: list[DecodedField] = []
        for index, layout in enumerate(self.fields):
            array = batch.column(index)
            validity = np.asarray(array.is_valid())
            expected_dtype = np.dtype(layout.dtype)
            if layout.shape == "scalar":
                values = np.asarray(array.to_numpy(zero_copy_only=False))
                if values.dtype != expected_dtype:
                    raise TypeError(
                        f"layout dtype mismatch for {layout.physical_name}: "
                        f"expected={expected_dtype} actual={values.dtype}"
                    )
                decoded.append(
                    DecodedField(layout.physical_name, values, validity=validity)
                )
            elif layout.shape == "fixed_list":
                if (
                    not pa.types.is_fixed_size_list(array.type)
                    or array.type.list_size != layout.width
                ):
                    raise TypeError(
                        f"fixed_list Arrow type mismatch for {layout.physical_name}"
                    )
                values = np.asarray(
                    array.values.to_numpy(zero_copy_only=False)
                ).reshape(batch.num_rows, layout.width)
                if values.dtype != expected_dtype:
                    raise TypeError(f"layout dtype mismatch for {layout.physical_name}")
                decoded.append(
                    DecodedField(layout.physical_name, values, validity=validity)
                )
            else:
                if not pa.types.is_list(array.type) and not pa.types.is_large_list(
                    array.type
                ):
                    raise TypeError(
                        f"ragged field is not a list array: {layout.physical_name}"
                    )
                offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False))
                values = np.asarray(array.values.to_numpy(zero_copy_only=False))
                if values.dtype != expected_dtype:
                    raise TypeError(f"layout dtype mismatch for {layout.physical_name}")
                decoded.append(
                    DecodedField(
                        layout.physical_name, values, offsets=offsets, validity=validity
                    )
                )
        return DecodedBatch(batch.num_rows, tuple(decoded))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "projection": list(self.projection),
            "fields": [
                {
                    "physical_name": field.physical_name,
                    "dtype": field.dtype,
                    "shape": field.shape,
                    "width": field.width,
                    "padding": field.padding,
                }
                for field in self.fields
            ],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BatchLayoutSpec:
        if set(value) != {"projection", "fields"}:
            raise ValueError("BatchLayoutSpec fields do not match protocol")
        fields = tuple(FieldLayout(**field) for field in value["fields"])
        return cls(
            fields=fields,
            projection=tuple(value["projection"]),
        )
