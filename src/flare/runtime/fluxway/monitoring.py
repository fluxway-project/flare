from __future__ import annotations

from typing import Any


async def wait_for_training_with_progress(execution: Any, interval_s: float) -> None:
    if interval_s <= 0:
        raise ValueError(f"interval_s must be positive, got {interval_s}")
    async for progress in execution.monitor_progress(interval_s):
        print(f"[Fluxway] Controller progress={progress}", flush=True)


__all__ = ["wait_for_training_with_progress"]
