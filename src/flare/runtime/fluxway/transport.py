"""Fluxway-owned transport wrapping and backend-native batch operations."""

from __future__ import annotations

from typing import Any

from flare.data.pipeline.consumption.source.batching import MappingBatchOperations
from flare.data.schema.batch import RaggedFeatureBatch
from fluxway.common.index_descriptor import IndexDescriptor
from fluxway.common.tensor_backend import get_backend
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor


def transport_payload(value: Any) -> Any:
    if not isinstance(value, TransportTensor):
        raise TypeError(
            f"transport_payload requires TransportTensor, got {type(value).__name__}"
        )
    return value.flat_values


def make_transport_tensor(flat_values: Any, **kwargs: Any) -> TransportTensor:
    return TransportTensor(flat_values=flat_values, **kwargs)


def transport_nested_splits(value: Any) -> list[Any]:
    if not isinstance(value, TransportTensor):
        raise TypeError(
            f"transport_nested_splits requires TransportTensor, got {type(value).__name__}"
        )
    return list(value.nested_splits)


def transport_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, TransportTensor):
        raise TypeError(
            f"transport_shape requires TransportTensor, got {type(value).__name__}"
        )
    return tuple(value.shape)


def transport_global_shape(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, TransportTensor):
        raise TypeError(
            f"transport_global_shape requires TransportTensor, got {type(value).__name__}"
        )
    return value.global_shape


def transport_alias(value: Any) -> TransportTensor:
    if not isinstance(value, TransportTensor):
        raise TypeError(
            f"transport_alias requires TransportTensor, got {type(value).__name__}"
        )
    return value.zero_copy_alias()


def transport_rebind(
    value: Any,
    *,
    origin_indices: Any,
    global_shape: tuple[int, ...] | None,
) -> TransportTensor:
    if not isinstance(value, TransportTensor):
        raise TypeError(
            f"transport_rebind requires TransportTensor, got {type(value).__name__}"
        )
    return TransportTensor(
        flat_values=value.flat_values,
        nested_splits=list(value.nested_splits),
        origin_indices=origin_indices,
        routing_indices=value.routing_indices,
        global_shape=global_shape,
        receipt=value.receipt,
        routing_axes=value.routing_axes,
        source_step=value.source_step,
    )


def wrap_ragged_feature_batch(
    value: RaggedFeatureBatch,
    *,
    global_rows: int,
    origin_indices: Any,
    source_step: int | None = None,
) -> TransportTensor:
    if not isinstance(value, RaggedFeatureBatch):
        raise TypeError("value must be a RaggedFeatureBatch")
    if type(global_rows) is not int or global_rows < value.row_count:
        raise ValueError("global_rows must cover the local ragged row count")
    return TransportTensor(
        flat_values=value.flat_values,
        nested_splits=list(value.nested_splits),
        origin_indices=origin_indices,
        global_shape=(global_rows, *value.feature_shape),
        source_step=source_step,
    )


class TransportTensorOperations:
    """Row operations that preserve transport metadata and backend residency."""

    @staticmethod
    def row_count(value: TransportTensor) -> int:
        backend = get_backend()
        if value.is_ragged:
            splits = value.nested_splits
            if not splits:
                raise ValueError("ragged transport tensor requires nested_splits")
            return int(backend.shape(splits[0])[0]) - 1
        shape = tuple(backend.shape(value.flat_values))
        return 1 if not shape else int(shape[0])

    @classmethod
    def _metadata(cls, values: list[TransportTensor]) -> dict[str, object]:
        shapes = tuple(value.global_shape for value in values)
        if any(shape is None or not shape for shape in shapes):
            raise ValueError(f"global_shape is required for every tensor: {shapes!r}")
        first_shape = shapes[0]
        if any(shape != first_shape for shape in shapes[1:]):
            raise ValueError(f"global_shape conflict: {shapes!r}")

        routing_axes = tuple(value.routing_axes for value in values)
        if any(axes != routing_axes[0] for axes in routing_axes[1:]):
            raise ValueError("transport tensors have incompatible routing_axes")
        if routing_axes[0] != (0,):
            raise ValueError("row batch operations require routing_axes=(0,)")

        source_steps = tuple(value.source_step for value in values)
        if any(step != source_steps[0] for step in source_steps[1:]):
            raise ValueError("transport tensors have incompatible source_step")

        result: dict[str, object] = {
            "global_shape": first_shape,
            "routing_axes": routing_axes[0],
            "source_step": source_steps[0],
        }
        for field in ("origin_indices", "routing_indices"):
            descriptors = [getattr(value, field) for value in values]
            if any(descriptor is not None for descriptor in descriptors):
                if any(descriptor is None for descriptor in descriptors):
                    raise ValueError(f"transport tensors have partial {field}")
                result[field] = IndexDescriptor.concatenate(descriptors)
            else:
                result[field] = None

        result["receipt"] = SymmetryReceipt.concatenate(
            [value.receipt for value in values],
            [cls.row_count(value) for value in values],
        )
        return result

    @staticmethod
    def _merge_splits(values: list[TransportTensor]) -> list[Any]:
        backend = get_backend()
        level_count = len(values[0].nested_splits)
        if any(len(value.nested_splits) != level_count for value in values[1:]):
            raise ValueError("ragged transport tensors have different nesting levels")
        merged: list[Any] = []
        for level in range(level_count):
            lengths = [
                splits[1:] - splits[:-1]
                for value in values
                for splits in (value.nested_splits[level],)
            ]
            merged_lengths = backend.cat(lengths, dim=0)
            zero = backend.zeros((1,), dtype=backend.dtype(merged_lengths))
            merged.append(
                backend.cat(
                    [zero, backend.cumsum(merged_lengths, dim=0)],
                    dim=0,
                )
            )
        return merged

    @classmethod
    def concatenate(cls, values: list[TransportTensor]) -> TransportTensor:
        if not values:
            raise ValueError("cannot concatenate an empty tensor list")
        if len(values) == 1:
            return values[0]
        if any(value.is_ragged != values[0].is_ragged for value in values[1:]):
            raise ValueError("cannot concatenate dense and ragged transport tensors")

        backend = get_backend()
        metadata = cls._metadata(values)
        payloads = [value.flat_values for value in values]
        shapes = [tuple(backend.shape(payload)) for payload in payloads]
        dtypes = [backend.dtype(payload) for payload in payloads]
        if any(dtype != dtypes[0] for dtype in dtypes[1:]):
            raise ValueError(f"payload dtype conflict: {dtypes!r}")
        if any(shape[1:] != shapes[0][1:] for shape in shapes[1:]):
            raise ValueError(f"payload trailing shape conflict: {shapes!r}")
        flat_values = backend.cat(payloads, dim=0)
        nested_splits = cls._merge_splits(values) if values[0].is_ragged else []
        return TransportTensor(
            flat_values=flat_values,
            nested_splits=nested_splits,
            origin_indices=metadata["origin_indices"],
            routing_indices=metadata["routing_indices"],
            global_shape=metadata["global_shape"],
            receipt=metadata["receipt"],
            routing_axes=metadata["routing_axes"],
            source_step=metadata["source_step"],
        )

    @staticmethod
    def take(value: TransportTensor, start: int, count: int) -> TransportTensor:
        return value.slice(start, start + count, unsafe=False)

    @classmethod
    def pad(cls, value: TransportTensor, count: int) -> TransportTensor:
        if count < 0:
            raise ValueError(f"padding count must be non-negative, got {count}")
        if count == 0:
            return value
        rows = cls.row_count(value)
        if rows == 0:
            raise ValueError("cannot pad an empty transport tensor")
        last = cls.take(value, rows - 1, 1)
        return cls.concatenate([value, *([last] * count)])


def create_transport_batch_operations() -> MappingBatchOperations:
    return MappingBatchOperations({TransportTensor: TransportTensorOperations()})


def transport_row_count(value: Any) -> int:
    if not isinstance(value, TransportTensor):
        raise TypeError(
            f"transport_row_count requires TransportTensor, got {type(value).__name__}"
        )
    return TransportTensorOperations.row_count(value)


__all__ = [
    "TransportTensorOperations",
    "create_transport_batch_operations",
    "make_transport_tensor",
    "transport_alias",
    "transport_global_shape",
    "transport_nested_splits",
    "transport_payload",
    "transport_rebind",
    "transport_row_count",
    "transport_shape",
    "wrap_ragged_feature_batch",
]
