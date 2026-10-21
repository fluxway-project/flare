from __future__ import annotations

import threading
import typing
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ReadProfileSnapshot:
    compressed_bytes: int
    fetch_wall_seconds: float
    fetch_cpu_seconds: float
    prefetch_wait_seconds: float
    csv_decode_seconds: float
    arrow_materialize_seconds: float
    yield_wait_seconds: float
    stream_setup_seconds: float
    csv_open_seconds: float
    open_seconds: float


@dataclass(slots=True)
class ReadProfile:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _compressed_bytes: int = 0
    _fetch_wall_seconds: float = 0.0
    _fetch_cpu_seconds: float = 0.0
    _prefetch_wait_seconds: float = 0.0
    _csv_decode_seconds: float = 0.0
    _arrow_materialize_seconds: float = 0.0
    _yield_wait_seconds: float = 0.0
    _stream_setup_seconds: float = 0.0
    _csv_open_seconds: float = 0.0
    _open_seconds: float = 0.0

    def record_fetch(self, size: int, wall_seconds: float, cpu_seconds: float) -> None:
        with self._lock:
            self._compressed_bytes += size
            self._fetch_wall_seconds += wall_seconds
            self._fetch_cpu_seconds += cpu_seconds

    def record_prefetch_wait(self, seconds: float) -> None:
        with self._lock:
            self._prefetch_wait_seconds += seconds

    def record_csv_decode(self, seconds: float) -> None:
        with self._lock:
            self._csv_decode_seconds += seconds

    def record_arrow_materialize(self, seconds: float) -> None:
        with self._lock:
            self._arrow_materialize_seconds += seconds

    def record_yield_wait(self, seconds: float) -> None:
        with self._lock:
            self._yield_wait_seconds += seconds

    def record_stream_setup(self, seconds: float) -> None:
        with self._lock:
            self._stream_setup_seconds += seconds

    def record_csv_open(self, seconds: float) -> None:
        with self._lock:
            self._csv_open_seconds += seconds

    def record_open(self, seconds: float) -> None:
        with self._lock:
            self._open_seconds += seconds

    def snapshot(self) -> ReadProfileSnapshot:
        with self._lock:
            return ReadProfileSnapshot(
                compressed_bytes=self._compressed_bytes,
                fetch_wall_seconds=self._fetch_wall_seconds,
                fetch_cpu_seconds=self._fetch_cpu_seconds,
                prefetch_wait_seconds=self._prefetch_wait_seconds,
                csv_decode_seconds=self._csv_decode_seconds,
                arrow_materialize_seconds=self._arrow_materialize_seconds,
                yield_wait_seconds=self._yield_wait_seconds,
                stream_setup_seconds=self._stream_setup_seconds,
                csv_open_seconds=self._csv_open_seconds,
                open_seconds=self._open_seconds,
            )


@dataclass(frozen=True, slots=True)
class ReadProfileSummary:
    files: int
    compressed_bytes: int
    total_seconds: float
    fetch_wall_seconds: float
    fetch_cpu_seconds: float
    prefetch_wait_seconds: float
    csv_decode_seconds: float
    arrow_materialize_seconds: float
    yield_wait_seconds: float
    stream_setup_seconds: float
    csv_open_seconds: float
    open_seconds: float


class _ReadProfileCollector:
    def __init__(self) -> None:
        self._profiles: list[tuple[ReadProfileSnapshot, float]] = []

    def record(self, snapshot: ReadProfileSnapshot, total_seconds: float) -> None:
        if total_seconds < 0.0:
            raise ValueError(f"total_seconds must be non-negative, got {total_seconds}")
        self._profiles.append((snapshot, total_seconds))

    def summary(self) -> ReadProfileSummary:
        if not self._profiles:
            raise RuntimeError("read profile collector received no file profiles")
        snapshots, totals = zip(*self._profiles, strict=True)
        return ReadProfileSummary(
            files=len(snapshots),
            compressed_bytes=sum(snapshot.compressed_bytes for snapshot in snapshots),
            total_seconds=sum(totals),
            fetch_wall_seconds=sum(
                snapshot.fetch_wall_seconds for snapshot in snapshots
            ),
            fetch_cpu_seconds=sum(snapshot.fetch_cpu_seconds for snapshot in snapshots),
            prefetch_wait_seconds=sum(
                snapshot.prefetch_wait_seconds for snapshot in snapshots
            ),
            csv_decode_seconds=sum(
                snapshot.csv_decode_seconds for snapshot in snapshots
            ),
            arrow_materialize_seconds=sum(
                snapshot.arrow_materialize_seconds for snapshot in snapshots
            ),
            yield_wait_seconds=sum(
                snapshot.yield_wait_seconds for snapshot in snapshots
            ),
            stream_setup_seconds=sum(
                snapshot.stream_setup_seconds for snapshot in snapshots
            ),
            csv_open_seconds=sum(snapshot.csv_open_seconds for snapshot in snapshots),
            open_seconds=sum(snapshot.open_seconds for snapshot in snapshots),
        )


def create_read_profile_collector() -> typing.Any:
    import ray

    return ray.remote(_ReadProfileCollector).options(num_cpus=0).remote()


def report_read_profile(
    collector: typing.Any,
    snapshot: ReadProfileSnapshot,
    total_seconds: float,
) -> None:
    collector.record.remote(snapshot, total_seconds)


def collect_read_profile_summary(collector: typing.Any) -> ReadProfileSummary:
    import ray

    summary = ray.get(collector.summary.remote())
    if not isinstance(summary, ReadProfileSummary):
        raise TypeError(
            f"unexpected read profile summary type: {type(summary).__name__}"
        )
    return summary
