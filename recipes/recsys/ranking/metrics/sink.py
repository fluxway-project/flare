from __future__ import annotations

from flare.training.components.metrics.contracts import MetricResult
from flare.training.components.metrics.sinks.tensorboard import TensorBoardMetricSink


class RankingTensorBoardMetricSink:
    """Adapt ranking metric names to the stable training dashboard schema."""

    def __init__(self, log_dir: str) -> None:
        self._sink = TensorBoardMetricSink(log_dir)

    def write(self, result: MetricResult) -> None:
        values: dict[str, float] = {}
        for name, value in result.values.items():
            if "/" not in name:
                values[f"train/{name}"] = float(value)
                continue
            target_name, metric_name = name.split("/", 1)
            if metric_name == "sample_count":
                metric_name = "metric_sample_count"
            values[f"train/{metric_name}_{target_name}"] = float(value)
        self._sink.write(MetricResult(values=values, metadata=result.metadata))

    def close(self) -> None:
        self._sink.close()


__all__ = ["RankingTensorBoardMetricSink"]
