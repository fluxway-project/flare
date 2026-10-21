from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from flare.data.pipeline.consumption.source.batching.contracts import (
    BatchValueOperations,
)
from flare.data.pipeline.consumption.source.contracts.model import SourceRecordRange


class MappingBatchOperations:
    def __init__(
        self, value_operations: Mapping[type, BatchValueOperations[Any]] | None = None
    ) -> None:
        self._value_operations = dict(value_operations or {})

    def _ops(self, value: Any) -> BatchValueOperations[Any]:
        for value_type, operations in self._value_operations.items():
            if isinstance(value, value_type):
                return operations
        if isinstance(value, np.ndarray):
            return NumpyOperations()
        raise TypeError(
            "unsupported batch value type; register BatchValueOperations explicitly: "
            f"{type(value).__name__}"
        )

    @staticmethod
    def _validate_keys(values: list[Mapping[str, Any]]) -> tuple[str, ...]:
        if not values:
            raise ValueError("cannot assemble an empty fragment list")
        expected = tuple(values[0].keys())
        if not expected:
            raise ValueError("batch fragments must contain at least one field")
        expected_set = set(expected)
        for value in values[1:]:
            if set(value) != expected_set:
                raise ValueError("batch fragments must have identical field sets")
        return expected

    def row_count(self, value: Mapping[str, Any]) -> int:
        keys = self._validate_keys([value])
        counts = {self._ops(value[key]).row_count(value[key]) for key in keys}
        if len(counts) != 1:
            raise ValueError(
                f"batch fields are not row-aligned: counts={sorted(counts)}"
            )
        return counts.pop()

    def concatenate(self, values: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        keys = self._validate_keys(values)
        return {
            key: self._ops(values[0][key]).concatenate([value[key] for value in values])
            for key in keys
        }

    def take(
        self, value: Mapping[str, Any], start: int, count: int
    ) -> Mapping[str, Any]:
        keys = self._validate_keys([value])
        return {
            key: self._ops(value[key]).take(value[key], start, count) for key in keys
        }

    def pad(self, value: Mapping[str, Any], count: int) -> Mapping[str, Any]:
        if count < 0:
            raise ValueError(f"padding count must be non-negative, got {count}")
        if count == 0:
            return value
        keys = self._validate_keys([value])
        return {key: self._ops(value[key]).pad(value[key], count) for key in keys}


class PyArrowTableOperations:
    def row_count(self, value: Any) -> int:
        return int(value.num_rows)

    def concatenate(self, values: list[Any]) -> Any:
        import pyarrow as pa

        if len(values) == 1:
            return values[0]
        return pa.concat_tables(values, promote_options="permissive")

    def take(self, value: Any, start: int, count: int) -> Any:
        return value.slice(start, count)

    def pad(self, value: Any, count: int) -> Any:
        if count < 0:
            raise ValueError(f"padding count must be non-negative, got {count}")
        if count == 0:
            return value
        if value.num_rows == 0:
            raise ValueError("cannot pad an empty table")
        import pyarrow as pa

        last_row = value.slice(value.num_rows - 1, 1)
        return pa.concat_tables(
            [value] + [last_row] * count,
            promote_options="permissive",
        )


class SourceRangeBatchOperations:
    def row_count(self, value: tuple[SourceRecordRange, ...]) -> int:
        return sum(item.record_count for item in value)

    def concatenate(
        self, values: list[tuple[SourceRecordRange, ...]]
    ) -> tuple[SourceRecordRange, ...]:
        return tuple(item for value in values for item in value)

    def take(
        self, value: tuple[SourceRecordRange, ...], start: int, count: int
    ) -> tuple[SourceRecordRange, ...]:
        rows = self.row_count(value)
        if start < 0 or count < 0 or start + count > rows:
            raise ValueError(
                "source range take exceeds payload: "
                f"start={start} count={count} rows={rows}"
            )
        if count == 0:
            return ()
        selected: list[SourceRecordRange] = []
        remaining_start = start
        remaining_count = count
        for item in value:
            if remaining_start >= item.record_count:
                remaining_start -= item.record_count
                continue
            take = min(item.record_count - remaining_start, remaining_count)
            selected.append(item.slice(remaining_start, take))
            remaining_count -= take
            remaining_start = 0
            if remaining_count == 0:
                break
        if remaining_count:
            raise RuntimeError("source range take did not cover the requested count")
        return tuple(selected)

    def pad(
        self, value: tuple[SourceRecordRange, ...], count: int
    ) -> tuple[SourceRecordRange, ...]:
        if count < 0:
            raise ValueError(f"padding count must be non-negative, got {count}")
        if count == 0:
            return value
        if not value:
            raise ValueError("cannot pad empty source ranges")
        last = value[-1].slice(value[-1].record_count - 1, 1)
        return (*value, *((last,) * count))


class NumpyOperations:
    def row_count(self, value: np.ndarray) -> int:
        if value.ndim == 0:
            raise ValueError("batch arrays must have a leading row dimension")
        return int(value.shape[0])

    def concatenate(self, values: list[np.ndarray]) -> np.ndarray:
        first = values[0]
        if len(values) == 1:
            return first
        for value in values[1:]:
            if value.dtype != first.dtype or value.shape[1:] != first.shape[1:]:
                raise ValueError("batch arrays must share dtype and trailing shape")
        return np.concatenate(values, axis=0)

    def take(self, value: np.ndarray, start: int, count: int) -> np.ndarray:
        return value[start : start + count]

    def pad(self, value: np.ndarray, count: int) -> np.ndarray:
        if value.shape[0] == 0:
            raise ValueError("cannot pad an empty batch")
        return np.concatenate((value, np.repeat(value[-1:], count, axis=0)), axis=0)
