from __future__ import annotations

import asyncio
import threading
from builtins import ExceptionGroup

import pyarrow as pa
import pytest

from flare.data.contracts.assets import AssetRef
from flare.data.contracts.lineage import (
    InputRead,
    LineageConfig,
    LineageContext,
    OutputPublished,
)
from flare.data.contracts.sink import SinkResult
from flare.data.pipeline.lineage import LineageRecorder
from flare.data.pipeline.production.publication.contract import PublicationIdentity


class Exporter:
    def __init__(self):
        self.batches = []
        self.closes = 0
        self.fail = False

    async def export(self, events):
        if self.fail:
            raise ValueError("delivery failed")
        self.batches.append(events)

    async def close(self):
        self.closes += 1


class Factory:
    def __init__(self, exporter):
        self.exporter = exporter
        self.contexts = []

    async def create(self, context):
        self.contexts.append(context)
        return self.exporter


def configuration(exporter, **kwargs):
    return LineageConfig(
        Factory(exporter), LineageContext("run", "operation"), **kwargs
    )


@pytest.mark.asyncio
async def test_read_facts_are_coalesced_without_data_or_per_batch_delivery():
    exporter = Exporter()
    config = configuration(exporter)
    recorder = LineageRecorder(config)
    asset = AssetRef("catalog", "dataset", "snapshot")
    for _ in range(100):
        await recorder.inputs_read((asset,))
    assert exporter.batches == []
    assert recorder.pending_events == (InputRead(asset),)
    await recorder.flush()
    await recorder.inputs_read((asset,))
    assert recorder.pending_events == ()
    await recorder.close()
    await recorder.close()
    assert exporter.batches == [(InputRead(asset),)]
    assert exporter.closes == 1
    assert config.exporter_factory.contexts == [config.context]


@pytest.mark.asyncio
async def test_duplicate_input_facts_do_not_schedule_child_tasks():
    exporter = Exporter()
    recorder = LineageRecorder(configuration(exporter))
    asset = AssetRef("catalog", "dataset", "snapshot")
    await recorder.inputs_read((asset,))
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    scheduled = []

    def task_factory(loop, coroutine, **options):
        scheduled.append(coroutine)
        if previous_factory is not None:
            return previous_factory(loop, coroutine, **options)
        return asyncio.Task(coroutine, loop=loop, **options)

    try:
        loop.set_task_factory(task_factory)
        for _ in range(100):
            await recorder.inputs_read((asset,))
        assert scheduled == []
    finally:
        loop.set_task_factory(previous_factory)
    await recorder.flush()
    try:
        loop.set_task_factory(task_factory)
        for _ in range(100):
            await recorder.inputs_read((asset,))
        assert scheduled == []
    finally:
        loop.set_task_factory(previous_factory)
        await recorder.close()


@pytest.mark.asyncio
async def test_duplicate_seen_inputs_preserve_recency_before_eviction():
    recorder = LineageRecorder(configuration(Exporter(), coalesce_window=2))
    first, second, third = [AssetRef("catalog", str(index)) for index in range(3)]
    for asset in (first, second):
        await recorder.inputs_read((asset,))
        await recorder.flush()
    await recorder.inputs_read((first,))
    await recorder.inputs_read((third,))
    await recorder.flush()
    await recorder.inputs_read((first,))
    assert recorder.pending_events == ()
    await recorder.inputs_read((second,))
    assert recorder.pending_events == (InputRead(second),)
    await recorder.close()


@pytest.mark.asyncio
async def test_pending_metadata_is_bounded_and_export_failures_retain_facts():
    exporter = Exporter()
    recorder = LineageRecorder(configuration(exporter, max_pending_events=2))
    assets = tuple(AssetRef("files", str(index)) for index in range(3))
    await recorder.inputs_read(assets)
    assert exporter.batches == [(InputRead(assets[0]), InputRead(assets[1]))]
    assert recorder.pending_events == (InputRead(assets[2]),)
    exporter.fail = True
    with pytest.raises(ValueError, match="delivery failed"):
        await recorder.flush()
    assert recorder.pending_events == (InputRead(assets[2]),)
    exporter.fail = False
    await recorder.close()
    assert exporter.batches[-1] == (InputRead(assets[2]),)


@pytest.mark.asyncio
async def test_native_async_export_does_not_block_other_coroutines():
    entered, release = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()

    class NativeExporter(Exporter):
        async def export(self, events):
            assert asyncio.get_running_loop() is loop
            entered.set()
            await release.wait()
            await super().export(events)

    exporter = NativeExporter()
    recorder = LineageRecorder(configuration(exporter))
    await recorder.inputs_read((AssetRef("query", "result"),))
    flushing = asyncio.create_task(recorder.flush())
    await asyncio.wait_for(entered.wait(), timeout=2)
    assert not flushing.done()
    await recorder.inputs_read((AssetRef("query", "other"),))
    release.set()
    await asyncio.wait_for(flushing, timeout=2)
    await recorder.close()
    assert sum(map(len, exporter.batches)) == 2


@pytest.mark.asyncio
async def test_sync_export_is_offloaded_and_close_releases_resources_after_failure():
    main_thread = threading.get_ident()

    class SyncExporter:
        closes = 0

        def export(self, events):
            assert threading.get_ident() != main_thread
            raise ValueError("delivery failed")

        def close(self):
            self.closes += 1
            raise RuntimeError("release failed")

    exporter = SyncExporter()
    recorder = LineageRecorder(configuration(exporter))
    await recorder.inputs_read((AssetRef("native", "dataset"),))
    with pytest.raises(ExceptionGroup, match="lineage close failed") as raised:
        await recorder.close()
    assert [type(error) for error in raised.value.exceptions] == [
        ValueError,
        RuntimeError,
    ]
    assert exporter.closes == 1
    assert len(recorder.pending_events) == 1


@pytest.mark.asyncio
async def test_publication_and_lineage_failure_keep_the_committed_result():
    exporter = Exporter()
    exporter.fail = True
    recorder = LineageRecorder(configuration(exporter, max_pending_events=1))
    await recorder.inputs_read((AssetRef("input", "dataset"),))
    asset = AssetRef("output", "dataset", "committed-version")
    result = SinkResult(2, (asset,))
    payload = object()
    publication = PublicationIdentity("publication")

    class NativeSink:
        calls = 0

        async def publish(self, value, *, schema, publication):
            assert value is payload
            self.calls += 1
            return result

    sink = NativeSink()
    committed = await sink.publish(
        payload, schema=pa.schema([]), publication=publication
    )
    with pytest.raises(ValueError, match="delivery failed"):
        await recorder.output_published(committed.outputs, publication.value)
    assert sink.calls == 1
    exporter.fail = False
    await recorder.output_published(committed.outputs, publication.value)
    await recorder.close()
    assert exporter.batches[-1] == (OutputPublished(asset, publication.value),)
    assert sink.calls == 1


@pytest.mark.asyncio
async def test_cancelled_close_retains_flush_and_single_exporter_release():
    entered, release = asyncio.Event(), asyncio.Event()

    class DelayedExporter(Exporter):
        async def export(self, events):
            entered.set()
            await release.wait()
            await super().export(events)

    exporter = DelayedExporter()
    recorder = LineageRecorder(configuration(exporter))
    asset = AssetRef("native", "dataset")
    await recorder.inputs_read((asset,))
    closing = asyncio.create_task(recorder.close())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        waiting = asyncio.create_task(recorder.close())
        await asyncio.sleep(0)
        assert not waiting.done()
        assert exporter.closes == 0
        with pytest.raises(RuntimeError, match="closed"):
            await recorder.inputs_read((asset,))
    finally:
        release.set()
        await recorder.close()
    await waiting
    assert exporter.batches == [(InputRead(asset),)]
    assert exporter.closes == 1


@pytest.mark.asyncio
async def test_cancelled_sync_flush_keeps_export_owned_before_close():
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    class SyncExporter:
        calls = 0
        running = False
        closes = 0

        def export(self, events):
            self.calls += 1
            self.running = True
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(timeout=5)
            self.running = False

        def close(self):
            assert not self.running
            self.closes += 1

    exporter = SyncExporter()
    recorder = LineageRecorder(configuration(exporter))
    await recorder.inputs_read((AssetRef("native", "dataset"),))
    flushing = asyncio.create_task(recorder.flush())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        flushing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flushing
        closing = asyncio.create_task(recorder.close())
        await asyncio.sleep(0)
        assert not closing.done()
        assert exporter.calls == 1
        assert exporter.closes == 0
    finally:
        release.set()
        await recorder.close()
    await closing
    assert exporter.calls == exporter.closes == 1


@pytest.mark.asyncio
async def test_exporter_cannot_close_its_recorder_reentrantly():
    class ReentrantExporter(Exporter):
        async def export(self, events):
            await recorder.close()

    exporter = ReentrantExporter()
    recorder = LineageRecorder(configuration(exporter))
    await recorder.inputs_read((AssetRef("native", "dataset"),))
    with pytest.raises(RuntimeError, match="cannot close its supervisor"):
        await asyncio.wait_for(recorder.flush(), timeout=2)
    exporter.export = Exporter.export.__get__(exporter)
    await recorder.close()


@pytest.mark.asyncio
async def test_parallel_recording_keeps_pending_bounded():
    exporter = Exporter()
    recorder = LineageRecorder(configuration(exporter, max_pending_events=2))
    assets = tuple(AssetRef("native", str(index)) for index in range(20))
    await asyncio.gather(*(recorder.inputs_read((asset,)) for asset in assets))
    assert len(recorder.pending_events) <= 2
    await recorder.close()
    exported = [event.asset for batch in exporter.batches for event in batch]
    assert set(exported) == set(assets)
    assert len(exported) == len(assets)


@pytest.mark.asyncio
async def test_unused_recorder_never_opens_an_exporter():
    class UnusedFactory:
        def create(self, context):
            raise AssertionError("empty recorder must not open a connection")

    recorder = LineageRecorder(
        LineageConfig(UnusedFactory(), LineageContext("run", "operation"))
    )
    await recorder.flush()
    await recorder.close()


@pytest.mark.asyncio
async def test_cancelled_sync_exporter_creation_remains_owned_until_close():
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    exporter = Exporter()

    class DelayedFactory:
        calls = 0

        def create(self, context):
            self.calls += 1
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(timeout=5)
            return exporter

    factory = DelayedFactory()
    recorder = LineageRecorder(
        LineageConfig(factory, LineageContext("run", "operation"))
    )
    assert factory.calls == 0
    asset = AssetRef("native", "dataset")
    await recorder.inputs_read((asset,))
    flushing = asyncio.create_task(recorder.flush())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        flushing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flushing
        closing = asyncio.create_task(recorder.close())
        await asyncio.sleep(0)
        assert not closing.done()
        assert exporter.closes == 0
    finally:
        release.set()
        await recorder.close()
    await closing
    assert exporter.closes == factory.calls == 1
    assert exporter.batches == [(InputRead(asset),)]
