from __future__ import annotations

import typing
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageProfileSnapshot:
    stage: str
    calls: int
    input_rows: int
    output_rows: int
    wall_seconds: float
    cpu_seconds: float


@dataclass(frozen=True, slots=True)
class StageProfileSummary:
    stage: str
    calls: int
    input_rows: int
    output_rows: int
    wall_seconds: float
    cpu_seconds: float


class _StageProfileCollector:
    def __init__(self) -> None:
        self._snapshots: list[StageProfileSnapshot] = []

    def record(self, snapshot: StageProfileSnapshot) -> None:
        if snapshot.calls != 1 or snapshot.wall_seconds < 0 or snapshot.cpu_seconds < 0:
            raise ValueError(f"invalid stage profile snapshot: {snapshot!r}")
        self._snapshots.append(snapshot)

    def summary(self, stage: str) -> StageProfileSummary:
        snapshots = [
            snapshot for snapshot in self._snapshots if snapshot.stage == stage
        ]
        if not snapshots:
            raise RuntimeError(f"stage profile collector received no {stage} snapshots")
        return StageProfileSummary(
            stage=stage,
            calls=len(snapshots),
            input_rows=sum(snapshot.input_rows for snapshot in snapshots),
            output_rows=sum(snapshot.output_rows for snapshot in snapshots),
            wall_seconds=sum(snapshot.wall_seconds for snapshot in snapshots),
            cpu_seconds=sum(snapshot.cpu_seconds for snapshot in snapshots),
        )


def create_stage_profile_collector() -> typing.Any:
    import ray

    return ray.remote(_StageProfileCollector).options(num_cpus=0).remote()


def report_stage_profile(collector: typing.Any, snapshot: StageProfileSnapshot) -> None:
    collector.record.remote(snapshot)


def collect_stage_profile_summary(
    collector: typing.Any, stage: str
) -> StageProfileSummary:
    import ray

    summary = ray.get(collector.summary.remote(stage))
    if not isinstance(summary, StageProfileSummary):
        raise TypeError(
            f"unexpected stage profile summary type: {type(summary).__name__}"
        )
    return summary
