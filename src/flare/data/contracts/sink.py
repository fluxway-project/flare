from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

import pyarrow as pa

from flare.data.pipeline.production.publication.contract import PublicationIdentity

PayloadT_contra = TypeVar("PayloadT_contra", contravariant=True)


@dataclass(frozen=True, slots=True)
class SinkResult:
    total_rows: int

    def __post_init__(self) -> None:
        if self.total_rows < 0:
            raise ValueError("sink total_rows must not be negative")


@runtime_checkable
class TransactionalSink(Protocol[PayloadT_contra]):
    def publish(
        self,
        payload: PayloadT_contra,
        *,
        schema: pa.Schema,
        publication: PublicationIdentity,
    ) -> SinkResult: ...


@runtime_checkable
class StreamingSink(Protocol[PayloadT_contra]):
    def consume(self, payload: PayloadT_contra, *, schema: pa.Schema) -> SinkResult: ...
