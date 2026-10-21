from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, TypeVar, cast

from flare.data.pipeline.consumption.source.batching.contracts import (
    AssembledBatch,
    BatchCompletionPolicy,
    BatchValueOperations,
)
from flare.data.pipeline.consumption.source.batching.validity import BatchValidity
from flare.data.pipeline.consumption.source.contracts.model import SourceRecordRange

T = TypeVar("T")


class BatchAssembler(Generic[T]):
    def __init__(
        self,
        batch_size: int,
        operations: BatchValueOperations[T],
        policy: BatchCompletionPolicy,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        self._operations = operations
        self._policy = policy
        self._pending: list[T] = []
        self.pending_rows = 0
        self._pending_ranges: list[SourceRecordRange] = []
        self._ranges_known = True
        self._replay_ranges: tuple[SourceRecordRange, ...] = ()
        self.discarded_rows = 0

    @property
    def pending_ranges(self) -> tuple[SourceRecordRange, ...]:
        if not self._ranges_known:
            return ()
        return tuple(self._pending_ranges)

    @property
    def replay_ranges(self) -> tuple[SourceRecordRange, ...]:
        return self._replay_ranges

    def append(
        self,
        fragment: T,
        source_ranges: tuple[SourceRecordRange, ...] | None = None,
    ) -> tuple[AssembledBatch[T], ...]:
        rows = self._operations.row_count(fragment)
        if rows <= 0:
            raise ValueError("empty fragments are not valid batch input")
        if source_ranges is None:
            if self.pending_rows or not self._pending_ranges:
                self._ranges_known = False
        else:
            if sum(item.record_count for item in source_ranges) != rows:
                raise ValueError(
                    "source ranges must cover exactly the appended fragment: "
                    f"ranges={sum(item.record_count for item in source_ranges)} "
                    f"rows={rows}"
                )
            if not self._ranges_known:
                raise ValueError(
                    "source range metadata cannot be attached after an opaque "
                    "fragment has entered the pending batch"
                )
            self._pending_ranges.extend(source_ranges)
        self._pending.append(fragment)
        self.pending_rows += rows
        return self.take_complete()

    def take_complete(self) -> tuple[AssembledBatch[T], ...]:
        output: list[AssembledBatch[T]] = []
        while self.pending_rows >= self.batch_size:
            payload = self._materialize_pending()
            batch = self._operations.take(payload, 0, self.batch_size)
            remainder = self._operations.take(
                payload, self.batch_size, self.pending_rows - self.batch_size
            )
            if self._ranges_known:
                self._consume_ranges(self.batch_size)
                remainder_ranges = self._pending_ranges
            else:
                remainder_ranges = []
            self._pending = [] if self.pending_rows == self.batch_size else [remainder]
            self._pending_ranges = (
                [] if self.pending_rows == self.batch_size else remainder_ranges
            )
            self.pending_rows -= self.batch_size
            if self.pending_rows == 0:
                self._ranges_known = True
            output.append(
                AssembledBatch(batch, BatchValidity(self.batch_size, self.batch_size))
            )
        return tuple(output)

    def finish_boundary(
        self, *, terminal: bool = False
    ) -> tuple[AssembledBatch[T], ...]:
        return cast(
            tuple[AssembledBatch[T], ...],
            self._policy.finalize(self, terminal=terminal),
        )

    def discard_pending(self, *, count_as_discarded: bool = False) -> None:
        if count_as_discarded:
            self.discarded_rows += self.pending_rows
        self._pending.clear()
        self.pending_rows = 0
        self._pending_ranges.clear()
        self._ranges_known = True

    def pad_pending(self) -> AssembledBatch[T]:
        if self.pending_rows == 0:
            raise ValueError("cannot pad an empty pending batch")
        payload = self._materialize_pending()
        samples_valid_count = self.pending_rows
        padded = self._operations.pad(payload, self.batch_size - samples_valid_count)
        self.discard_pending()
        return AssembledBatch(
            padded, BatchValidity(self.batch_size, samples_valid_count)
        )

    def _materialize_pending(self) -> T:
        if not self._pending:
            raise ValueError("cannot materialize an empty pending batch")
        return self._operations.concatenate(self._pending)

    def _consume_ranges(self, count: int) -> tuple[SourceRecordRange, ...]:
        if count <= 0:
            raise ValueError(f"range consumption count must be positive, got {count}")
        if not self._ranges_known:
            return ()
        if sum(item.record_count for item in self._pending_ranges) < count:
            raise RuntimeError("source range metadata is shorter than pending rows")
        consumed: list[SourceRecordRange] = []
        remaining = count
        pending = list(self._pending_ranges)
        while remaining:
            item = pending.pop(0)
            take = min(item.record_count, remaining)
            consumed.append(item.slice(0, take))
            remaining -= take
            if take < item.record_count:
                pending.insert(0, item.slice(take, item.record_count - take))
        self._pending_ranges = pending
        return tuple(consumed)

    def checkpoint_state(self) -> dict[str, Any]:
        if self.pending_rows and not self._ranges_known:
            raise RuntimeError(
                "cannot checkpoint an incomplete batch without source range metadata"
            )
        return {
            "pending_rows": self.pending_rows,
            "discarded_rows": self.discarded_rows,
            "pending_ranges": [item.to_dict() for item in self.pending_ranges],
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"pending_rows", "discarded_rows", "pending_ranges"}:
            raise ValueError("assembler checkpoint fields do not match protocol")
        pending_rows = state.get("pending_rows")
        discarded_rows = state.get("discarded_rows")
        raw_ranges = state.get("pending_ranges")
        if (
            not isinstance(pending_rows, int)
            or isinstance(pending_rows, bool)
            or pending_rows < 0
        ):
            raise ValueError("assembler pending_rows must be a non-negative integer")
        if (
            not isinstance(discarded_rows, int)
            or isinstance(discarded_rows, bool)
            or discarded_rows < 0
        ):
            raise ValueError("assembler discarded_rows must be a non-negative integer")
        if not isinstance(raw_ranges, list):
            raise TypeError("assembler pending_ranges must be a list")
        ranges = tuple(SourceRecordRange.from_mapping(item) for item in raw_ranges)
        if sum(item.record_count for item in ranges) != pending_rows:
            raise ValueError("assembler pending_ranges do not cover pending_rows")
        self._pending.clear()
        self.pending_rows = 0
        self._pending_ranges.clear()
        self._ranges_known = True
        self._replay_ranges = ranges
        self.discarded_rows = discarded_rows
