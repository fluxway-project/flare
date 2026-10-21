from __future__ import annotations

from flare.data.pipeline.production.publish.read_profile import (
    ReadProfileSnapshot,
    report_read_profile,
)
from flare.data.pipeline.production.publish.stage_profile import (
    StageProfileSnapshot,
    report_stage_profile,
)


class _RemoteMethod:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def remote(self, *args: object) -> object:
        self.calls.append(args)
        return object()


class _Collector:
    def __init__(self) -> None:
        self.record = _RemoteMethod()


def test_report_read_profile_submits_without_waiting() -> None:
    collector = _Collector()
    snapshot = ReadProfileSnapshot(
        compressed_bytes=1,
        fetch_wall_seconds=0.0,
        fetch_cpu_seconds=0.0,
        prefetch_wait_seconds=0.0,
        csv_decode_seconds=0.0,
        arrow_materialize_seconds=0.0,
        yield_wait_seconds=0.0,
        stream_setup_seconds=0.0,
        csv_open_seconds=0.0,
        open_seconds=0.0,
    )

    result = report_read_profile(collector, snapshot, 1.0)

    assert result is None
    assert collector.record.calls == [(snapshot, 1.0)]


def test_report_stage_profile_submits_without_waiting() -> None:
    collector = _Collector()
    snapshot = StageProfileSnapshot(
        stage="encode",
        calls=1,
        input_rows=1,
        output_rows=1,
        wall_seconds=0.0,
        cpu_seconds=0.0,
    )

    result = report_stage_profile(collector, snapshot)

    assert result is None
    assert collector.record.calls == [(snapshot,)]
