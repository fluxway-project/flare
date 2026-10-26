"""The standard file-range implementation's storage reader contract."""

from collections.abc import AsyncIterator, Awaitable, Sequence
from typing import Any, Protocol, runtime_checkable

# pyrefly: ignore[missing-import]
import pyarrow as pa

from flare.data.sources.range import SourceRecordRange


@runtime_checkable
class RecordRangeReader(Protocol):
    @property
    def schema(self) -> pa.Schema: ...

    def bind_source_identity(self, source_identity: str) -> None: ...

    def close(self) -> None | Awaitable[None]: ...

    def read_batches(
        self,
        record_range: SourceRecordRange,
        projection: Sequence[str],
        *,
        batch_size: int,
    ) -> AsyncIterator[Any] | object | Awaitable[object]: ...

    def read(
        self, record_range: SourceRecordRange, projection: Sequence[str]
    ) -> object | Awaitable[object]: ...
