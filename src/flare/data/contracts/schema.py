from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa


@dataclass(frozen=True, slots=True)
class SchemaContract:
    input_schema: pa.Schema
    output_schema: pa.Schema

    def __post_init__(self) -> None:
        if not isinstance(self.input_schema, pa.Schema):
            raise TypeError("input_schema must be a PyArrow schema")
        if not isinstance(self.output_schema, pa.Schema):
            raise TypeError("output_schema must be a PyArrow schema")

    def validate_input(self, batch: pa.Table) -> None:
        self._validate(batch, self.input_schema, "input")

    def validate_output(self, batch: pa.Table) -> None:
        self._validate(batch, self.output_schema, "output")

    @staticmethod
    def _validate(batch: pa.Table, expected: pa.Schema, boundary: str) -> None:
        if not isinstance(batch, pa.Table):
            raise TypeError(f"{boundary} batch must be a PyArrow table")
        if not batch.schema.equals(expected, check_metadata=True):
            raise ValueError(
                f"{boundary} batch schema mismatch: "
                f"expected={expected!s} actual={batch.schema!s}"
            )
