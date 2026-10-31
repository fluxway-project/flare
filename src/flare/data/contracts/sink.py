from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

import pyarrow as pa

from flare.data.contracts import publication as publication_contracts
from flare.data.contracts.assets import AssetRef

PayloadT_contra = TypeVar("PayloadT_contra", contravariant=True)


@dataclass(frozen=True, slots=True)
class SinkResult:
    total_rows: int
    outputs: tuple[AssetRef, ...] = ()

    def __post_init__(self) -> None:
        if self.total_rows < 0:
            raise ValueError("sink total_rows must not be negative")
        if not isinstance(self.outputs, tuple) or any(
            not isinstance(asset, AssetRef) for asset in self.outputs
        ):
            raise TypeError("sink outputs must be an AssetRef tuple")


@runtime_checkable
class TransactionalSink(Protocol[PayloadT_contra]):
    def publish(
        self,
        payload: PayloadT_contra,
        *,
        schema: pa.Schema,
        publication: publication_contracts.PublicationIdentity,
    ) -> SinkResult | Awaitable[SinkResult]: ...


@runtime_checkable
class StreamingSink(Protocol[PayloadT_contra]):
    def consume(self, payload: PayloadT_contra, *, schema: pa.Schema) -> SinkResult: ...


__all__ = ["SinkResult", "StreamingSink", "TransactionalSink"]
