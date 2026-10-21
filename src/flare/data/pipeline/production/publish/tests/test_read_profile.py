from __future__ import annotations

import pytest

from flare.data.pipeline.production.publish.read_profile import (
    ReadProfileSnapshot,
    _ReadProfileCollector,
)


def _snapshot(compressed_bytes: int) -> ReadProfileSnapshot:
    return ReadProfileSnapshot(
        compressed_bytes=compressed_bytes,
        fetch_wall_seconds=1.0,
        fetch_cpu_seconds=0.2,
        prefetch_wait_seconds=0.3,
        csv_decode_seconds=0.4,
        arrow_materialize_seconds=0.1,
        yield_wait_seconds=0.6,
        stream_setup_seconds=0.7,
        csv_open_seconds=0.8,
        open_seconds=0.5,
    )


def test_read_profile_collector_aggregates_all_file_metrics():
    collector = _ReadProfileCollector()
    collector.record(_snapshot(7), 2.0)
    collector.record(_snapshot(11), 3.0)

    summary = collector.summary()

    assert summary.files == 2
    assert summary.compressed_bytes == 18
    assert summary.total_seconds == 5.0
    assert summary.fetch_wall_seconds == 2.0
    assert summary.prefetch_wait_seconds == 0.6
    assert summary.yield_wait_seconds == 1.2
    assert summary.stream_setup_seconds == 1.4
    assert summary.csv_open_seconds == 1.6


def test_read_profile_collector_rejects_invalid_elapsed_time():
    collector = _ReadProfileCollector()

    with pytest.raises(ValueError, match="total_seconds"):
        collector.record(_snapshot(7), -0.1)
