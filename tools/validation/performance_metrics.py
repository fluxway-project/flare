"""Small, dependency-free timing and throughput calculations for benchmarks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class Measurement:
    elapsed_seconds: float
    rows: int

    def __post_init__(self) -> None:
        if self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive")
        if self.rows < 0:
            raise ValueError("rows must not be negative")

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.elapsed_seconds


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(measurements: Iterable[Measurement]) -> dict[str, Any]:
    values = tuple(measurements)
    if not values:
        raise ValueError("summarize requires at least one measurement")
    elapsed = [item.elapsed_seconds for item in values]
    throughput = [item.rows_per_second for item in values]
    return {
        "repetitions": len(values),
        "rows": values[0].rows,
        "elapsed_seconds": {
            "median": median(elapsed),
            "p50": percentile(elapsed, 0.50),
            "p95": percentile(elapsed, 0.95),
        },
        "rows_per_second": {
            "median": median(throughput),
            "p50": percentile(throughput, 0.50),
            "p95": percentile(throughput, 0.95),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = ["Measurement", "percentile", "summarize", "write_json"]
