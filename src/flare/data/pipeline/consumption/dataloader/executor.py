"""Persistent asyncio/threads executor for bounded RecordBatch streaming."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from flare.data.pipeline.consumption.source.contracts.model import SourceRecordRange


@dataclass(frozen=True, slots=True)
class _ReadFailure:
    error: Exception


class FormatAdapterRegistry:
    """Immutable format-id dispatch table injected at the connector seam."""

    def __init__(
        self, adapters: dict[str, Callable[..., Iterable[pa.RecordBatch]]]
    ) -> None:
        if not adapters:
            raise ValueError("format adapter registry must not be empty")
        normalized: dict[str, Callable[..., Iterable[pa.RecordBatch]]] = {}
        for format_id, adapter in adapters.items():
            if (
                not isinstance(format_id, str)
                or not format_id
                or format_id.strip() != format_id
            ):
                raise ValueError("format adapter id must be canonical")
            if not callable(adapter):
                raise TypeError(f"format adapter is not callable: {format_id!r}")
            if format_id in normalized:
                raise ValueError(f"duplicate format adapter: {format_id}")
            normalized[format_id] = adapter
        self._adapters = normalized

    def adapter_for(self, format_id: str) -> Callable[..., Iterable[pa.RecordBatch]]:
        try:
            return self._adapters[format_id]
        except KeyError as exc:
            raise ValueError(
                f"no adapter registered for format: {format_id!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ReadLimits:
    max_manifests: int = 2
    max_payload_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_manifests <= 0 or self.max_payload_bytes <= 0:
            raise ValueError("read limits must be positive")


class AsyncReadSession:
    """One persistent thread pool and event-loop bridge per runtime worker."""

    def __init__(
        self,
        *,
        registry: FormatAdapterRegistry,
        max_workers: int = 2,
        queue_capacity: int = 2,
    ) -> None:
        if max_workers <= 0 or queue_capacity <= 0:
            raise ValueError("AsyncReadSession capacities must be positive")
        self._registry = registry
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._queue_capacity = queue_capacity
        self._closed = False
        self._futures: set[concurrent.futures.Future[Any]] = set()

    async def read_batches(
        self,
        source_range: SourceRecordRange,
        projection: Sequence[str],
    ) -> AsyncIterator[pa.RecordBatch]:
        if self._closed:
            raise RuntimeError("AsyncReadSession is closed")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue(self._queue_capacity)
        end = object()
        stop = threading.Event()

        def produce() -> None:
            def put(value: object) -> bool:
                while not stop.is_set() and not self._closed:
                    pending = asyncio.run_coroutine_threadsafe(queue.put(value), loop)
                    try:
                        pending.result(timeout=0.1)
                        return True
                    except concurrent.futures.TimeoutError:
                        pending.cancel()
                return False

            try:
                adapter = self._registry.adapter_for(source_range.record_format)
                batches = adapter(source_range, tuple(projection))
                if inspect.isawaitable(batches):
                    raise TypeError(
                        "format adapter must return a synchronous batch iterator"
                    )
                for batch in batches:
                    if stop.is_set() or self._closed:
                        return
                    if not isinstance(batch, pa.RecordBatch):
                        raise TypeError("format adapter must yield pyarrow.RecordBatch")
                    if not put(batch):
                        return
                put(end)
            except Exception as error:  # noqa: BLE001 - worker must report adapter failures
                put(_ReadFailure(error))

        future = self._executor.submit(produce)
        self._futures.add(future)
        try:
            while True:
                item = await queue.get()
                if item is end:
                    return
                if isinstance(item, _ReadFailure):
                    raise item.error
                yield item
        finally:
            stop.set()
            future.cancel()
            self._futures.discard(future)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for future in tuple(self._futures):
            future.cancel()
        await asyncio.gather(
            *(asyncio.wrap_future(future) for future in tuple(self._futures)),
            return_exceptions=True,
        )
        self._executor.shutdown(wait=True, cancel_futures=True)


class AsyncBatchExecutor:
    """Double-bounded manifest/payload orchestration.

    ``materialize`` is called for each manifest.  The executor never acks a
    source position; ownership of durable progress remains in Fluxway.
    """

    def __init__(self, *, limits: ReadLimits) -> None:
        self._limits = limits
        self._manifest_slots = asyncio.Semaphore(limits.max_manifests)
        self._bytes_in_flight = 0
        self._bytes_condition = asyncio.Condition()
        self._closed = False

    async def submit(
        self,
        materialize: Callable[[], Any],
        *,
        estimated_bytes: int,
    ) -> Any:
        if self._closed:
            raise RuntimeError("AsyncBatchExecutor is closed")
        if (
            not isinstance(estimated_bytes, int)
            or isinstance(estimated_bytes, bool)
            or estimated_bytes <= 0
        ):
            raise ValueError("estimated_bytes must be positive")
        if estimated_bytes > self._limits.max_payload_bytes:
            raise ValueError("estimated payload exceeds executor byte budget")
        await self._manifest_slots.acquire()
        try:
            async with self._bytes_condition:
                await self._bytes_condition.wait_for(
                    lambda: (
                        self._bytes_in_flight + estimated_bytes
                        <= self._limits.max_payload_bytes
                    )
                )
                self._bytes_in_flight += estimated_bytes
            try:
                if inspect.iscoroutinefunction(materialize):
                    result = materialize()
                else:
                    result = await asyncio.to_thread(materialize)
                if inspect.isawaitable(result):
                    result = await result
                return result
            finally:
                async with self._bytes_condition:
                    self._bytes_in_flight -= estimated_bytes
                    self._bytes_condition.notify_all()
        finally:
            self._manifest_slots.release()

    async def close(self) -> None:
        self._closed = True
