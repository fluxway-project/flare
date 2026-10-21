from __future__ import annotations

import statistics
from collections.abc import Iterable


def median_duration_seconds(durations: Iterable[float]) -> float:
    values = tuple(float(duration) for duration in durations)
    if not values:
        raise ValueError("at least one duration is required")
    if any(duration <= 0.0 for duration in values):
        raise ValueError("durations must be positive")
    return float(statistics.median(values))


__all__ = ["median_duration_seconds"]
