from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pyarrow as pa

DEFAULT_VALIDITY_COLUMN: str = "samples_valid_count"
VirtualPadBatchFactory = Callable[[pa.Schema, Sequence[str], int], pa.RecordBatch]


class ArrowVirtualPadBatchFactory:
    """Factory to generate synthetic virtual pad batches from PyArrow schemas with explicit pad values."""

    @staticmethod
    def create_batch(
        schema: pa.Schema,
        projection: Sequence[str],
        count: int,
        pad_values: Mapping[str, Any] | None = None,
        *,
        validity_column: str | None = DEFAULT_VALIDITY_COLUMN,
    ) -> pa.RecordBatch:
        if not isinstance(schema, pa.Schema):
            raise TypeError("Virtual pad batch requires an explicit Arrow schema")
        if type(count) is not int or count <= 0:
            raise ValueError(f"Virtual pad batch count must be positive, got {count}")
        if not projection or len(set(projection)) != len(projection):
            raise ValueError("Virtual pad projection must be non-empty and unique")
        missing = set(projection).difference(schema.names)
        if missing:
            raise ValueError(
                f"Virtual pad schema is missing projected columns: {sorted(missing)!r}"
            )
        pad_map = dict(pad_values or {})
        arrays: list[pa.Array] = []
        names: list[str] = []

        for name in projection:
            field = schema.field(name)
            field_type = field.type
            is_nullable = field.nullable

            if name in pad_map:
                val = pad_map[name]
                if val is None and not is_nullable:
                    raise ValueError(
                        f"Virtual pad value for non-nullable column {name!r} cannot be null"
                    )
                arr = pa.array([val] * count, type=field_type)
            elif validity_column is not None and name == validity_column:
                arr = pa.array([0] * count, type=field_type)
            elif is_nullable:
                arr = pa.nulls(count, type=field_type)
            else:
                raise ValueError(
                    f"Virtual pad batch generation requires explicit pad value for non-nullable column {name!r}; "
                    "no default value assumed to protect model training integrity"
                )
            arrays.append(arr)
            names.append(name)

        return pa.RecordBatch.from_arrays(
            arrays,
            schema=pa.schema(
                [schema.field(name) for name in names], metadata=schema.metadata
            ),
        )


__all__ = [
    "DEFAULT_VALIDITY_COLUMN",
    "ArrowVirtualPadBatchFactory",
    "VirtualPadBatchFactory",
]
