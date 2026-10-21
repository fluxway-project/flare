from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from flare.data.pipeline.consumption.source.batching.validity import BatchValidity

T = TypeVar("T")


class BatchValueOperations(Protocol[T]):
    def row_count(self, value: T) -> int: ...

    def concatenate(self, values: list[T]) -> T: ...

    def take(self, value: T, start: int, count: int) -> T: ...

    def pad(self, value: T, count: int) -> T: ...


@dataclass(frozen=True, slots=True)
class AssembledBatch(Generic[T]):
    payload: T
    validity: BatchValidity


class CompletionContext(Protocol):
    batch_size: int
    pending_rows: int

    def discard_pending(self, *, count_as_discarded: bool = False) -> None: ...

    def pad_pending(self) -> AssembledBatch[Any]: ...


@runtime_checkable
class BatchCompletionPolicy(Protocol):
    def finalize(
        self,
        context: CompletionContext,
        *,
        terminal: bool,
    ) -> tuple[AssembledBatch[Any], ...]: ...
