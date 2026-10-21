import pytest

from flare.data.pipeline.production.publish.stage_profile import (
    StageProfileSnapshot,
    _StageProfileCollector,
)


def test_stage_profile_collector_aggregates_per_call_metrics() -> None:
    collector = _StageProfileCollector()
    collector.record(StageProfileSnapshot("parse", 1, 3, 3, 0.2, 0.1))
    collector.record(StageProfileSnapshot("parse", 1, 4, 4, 0.3, 0.2))

    summary = collector.summary("parse")

    assert summary.stage == "parse"
    assert summary.calls == 2
    assert summary.input_rows == 7
    assert summary.output_rows == 7
    assert summary.wall_seconds == pytest.approx(0.5)
    assert summary.cpu_seconds == pytest.approx(0.3)
