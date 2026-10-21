"""Model-payload stream reader built on the shared batch assembler."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pyarrow as pa

from flare.data.pipeline.consumption.source.batching import (
    BatchAssembler,
    completion_policy_for_name,
)
from flare.data.pipeline.consumption.source.batching.contracts import (
    BatchValueOperations,
)
from flare.data.pipeline.consumption.source.contracts.model import SourceRecordRange


def batch_validity_count(validity) -> np.ndarray:
    return np.asarray([validity.samples_valid_count], dtype=np.int32)


class StreamingBatchReader:
    """Keeps the assembler alive while source loaders are replaced."""

    def __init__(
        self,
        *,
        batch_size: int,
        completion_policy: str,
        operations: BatchValueOperations[Any],
        validity_field: str | None,
    ) -> None:
        self._assembler = BatchAssembler(
            batch_size,
            operations,
            completion_policy_for_name(completion_policy),
        )
        if validity_field is not None and (
            not validity_field or validity_field.strip() != validity_field
        ):
            raise ValueError("validity_field must be a non-empty canonical string")
        self._validity_field = validity_field

    @property
    def pending_rows(self) -> int:
        return self._assembler.pending_rows

    @property
    def replay_ranges(self) -> tuple[SourceRecordRange, ...]:
        return self._assembler.replay_ranges

    def checkpoint_state(self) -> dict[str, Any]:
        """Return source metadata while intentionally excluding local payload."""

        state = self._assembler.checkpoint_state()
        return state

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self._assembler.restore_checkpoint_state(state)

    def append_loader(
        self,
        loader: Iterator[Any],
    ) -> Iterator[Any]:
        for item in loader:
            if isinstance(item, tuple):
                if len(item) != 2 or not isinstance(item[0], (Mapping, pa.Table)):
                    raise TypeError(
                        "source loader tuple items must be (payload, source_ranges)"
                    )
                fragment, fragment_ranges = item
            else:
                fragment, fragment_ranges = item, None
            yield from (
                self._payload_with_validity(assembled)
                for assembled in self._assembler.append(fragment, fragment_ranges)
            )

    def append_fragment(
        self,
        fragment: Any,
        source_ranges: tuple[SourceRecordRange, ...],
    ) -> tuple[Any, ...]:
        return tuple(
            self._payload_with_validity(assembled)
            for assembled in self._assembler.append(fragment, source_ranges)
        )

    def finish_loader(self) -> tuple[Any, ...]:
        return tuple(
            self._payload_with_validity(assembled)
            for assembled in self._assembler.finish_boundary(terminal=False)
        )

    def finalize(self) -> tuple[Any, ...]:
        return tuple(
            self._payload_with_validity(assembled)
            for assembled in self._assembler.finish_boundary(terminal=True)
        )

    def _payload_with_validity(self, assembled: Any) -> Any:
        if isinstance(assembled.payload, pa.Table):
            if self._validity_field is None:
                return assembled.payload
            meta = dict(assembled.payload.schema.metadata or {})
            meta[self._validity_field.encode("utf-8")] = str(
                assembled.validity.samples_valid_count
            ).encode("utf-8")
            return assembled.payload.replace_schema_metadata(meta)
        payload = dict(assembled.payload)
        if self._validity_field is not None:
            payload[self._validity_field] = batch_validity_count(assembled.validity)
        return payload


def finite_batches(
    loader: Iterator[Any],
    *,
    batch_size: int,
    completion_policy: str,
    operations: BatchValueOperations[Any],
    validity_field: str | None,
) -> Iterator[Any]:
    reader = StreamingBatchReader(
        batch_size=batch_size,
        completion_policy=completion_policy,
        operations=operations,
        validity_field=validity_field,
    )
    yield from reader.append_loader(loader)
    yield from reader.finalize()
