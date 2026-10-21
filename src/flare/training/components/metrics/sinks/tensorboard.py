from __future__ import annotations

from pathlib import Path
from typing import Any

from flare.training.components.metrics.contracts import MetricResult, MetricSink


class TensorBoardMetricSink(MetricSink):
    """Sink that logs metric results to TensorBoard summary writer."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Any = None

    def write(self, result: MetricResult) -> None:
        if self._writer is None:
            from tensorboardX import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.log_dir))

        step = result.metadata["global_step"]
        if not isinstance(step, int) or step < 0:
            raise ValueError(
                f"metric metadata global_step must be a non-negative int, got {step!r}"
            )
        for name, value in result.values.items():
            self._writer.add_scalar(name, float(value), step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()


class ConsoleMetricSink(MetricSink):
    """Sink that logs metric results to standard console stdout."""

    def write(self, result: MetricResult) -> None:
        step = result.metadata["global_step"]
        if not isinstance(step, int) or step < 0:
            raise ValueError(
                f"metric metadata global_step must be a non-negative int, got {step!r}"
            )
        metrics_str = ", ".join(
            f"{name}={value:.4f}" for name, value in sorted(result.values.items())
        )
        print(f"[Step {step}] Metrics: {metrics_str}", flush=True)

    def close(self) -> None:
        return None


__all__ = ["ConsoleMetricSink", "TensorBoardMetricSink"]
