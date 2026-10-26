"""Bounded process-local facts with explicit asynchronous delivery."""

from __future__ import annotations

import asyncio
from builtins import BaseExceptionGroup
from collections import OrderedDict
from collections.abc import Callable, Sequence

from fluxway.common.async_close import AsyncCloseOnce
from fluxway.common.async_operations import AsyncOperationSupervisor

from flare.data.contracts.assets import AssetRef
from flare.data.contracts.lineage import (
    InputRead,
    LineageConfig,
    LineageEvent,
    LineageExporter,
    OutputPublished,
)
from flare.data.pipeline.consumption.source.execution.invocation import invoke_source


class LineageRecorder:
    def __init__(self, config: LineageConfig) -> None:
        if not isinstance(config, LineageConfig):
            raise TypeError("lineage recorder requires LineageConfig")
        self._config = config
        self._exporter: LineageExporter | None = None
        self._capacity = config.max_pending_events
        self._window = config.coalesce_window
        self._pending: OrderedDict[tuple[object, ...], LineageEvent] = OrderedDict()
        self._seen: OrderedDict[tuple[object, ...], None] = OrderedDict()
        self._flush_lock = asyncio.Lock()
        self._operations = AsyncOperationSupervisor()
        self._close_once = AsyncCloseOnce(self._finish_close)

    @property
    def pending_events(self) -> tuple[LineageEvent, ...]:
        return tuple(self._pending.values())

    def _coalesce_known(self, key: tuple[object, ...]) -> bool:
        if key in self._seen:
            self._seen.move_to_end(key)
            return True
        return key in self._pending

    async def _record(
        self,
        key: tuple[object, ...],
        create: Callable[..., LineageEvent],
        *args: object,
    ) -> None:
        if self._coalesce_known(key):
            return
        while len(self._pending) >= self._capacity:
            await self._flush_pending()
            if key in self._seen or key in self._pending:
                return
        self._pending[key] = create(*args)

    async def inputs_read(self, assets: Sequence[AssetRef]) -> None:
        if not self._operations.accepting:
            raise RuntimeError("lineage recorder is closed")
        # A fixed source snapshot repeats on every batch; known facts need no task.
        if all(self._coalesce_known((InputRead, asset)) for asset in assets):
            return
        await self._operations.run(lambda: self._inputs_read(assets))

    async def _inputs_read(self, assets: Sequence[AssetRef]) -> None:
        for asset in assets:
            await self._record((InputRead, asset), InputRead, asset)

    async def output_published(
        self, assets: Sequence[AssetRef], publication_id: str
    ) -> None:
        if not self._operations.accepting:
            raise RuntimeError("lineage recorder is closed")
        await self._operations.run(
            lambda: self._output_published(assets, publication_id)
        )

    async def _output_published(
        self, assets: Sequence[AssetRef], publication_id: str
    ) -> None:
        for asset in assets:
            await self._record(
                (OutputPublished, asset, publication_id),
                OutputPublished,
                asset,
                publication_id,
            )

    async def _flush_pending(self) -> None:
        async with self._flush_lock:
            pending = tuple(self._pending.items())
            if not pending:
                return
            if self._exporter is None:
                exporter = await invoke_source(
                    self._config.exporter_factory.create, self._config.context
                )
                if not isinstance(exporter, LineageExporter):
                    raise TypeError("lineage factory must create LineageExporter")
                self._exporter = exporter
            await invoke_source(
                self._exporter.export, tuple(event for _, event in pending)
            )
            for key, _event in pending:
                self._pending.pop(key)
                self._seen[key] = None
                self._seen.move_to_end(key)
                if len(self._seen) > self._window:
                    self._seen.popitem(last=False)

    async def flush(self) -> None:
        if not self._operations.accepting:
            raise RuntimeError("lineage recorder is closed")
        await self._operations.run(self._flush_pending)

    async def _finish_close(self) -> None:
        failures = list(await self._operations.drain())
        try:
            await self._flush_pending()
        except BaseException as error:  # noqa: BLE001
            failures.append(error)
        if self._exporter is not None:
            try:
                await invoke_source(self._exporter.close)
            except BaseException as error:  # noqa: BLE001
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("lineage close failed", failures)

    async def close(self) -> None:
        self._operations.stop_accepting()
        await self._close_once.close()
