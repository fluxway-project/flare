from dataclasses import dataclass
from typing import Any

from flare.training.components.metrics.contracts import MetricSpec


@dataclass(frozen=True)
class RankingMetricsConfig:
    target_names: tuple[str, ...]
    target_labels: Any
    window_steps: int = 100
    negative_factor: float | None = None
    max_inflight_steps: int = 4


def build_auc_metric_specs(target_names: tuple[str, ...]) -> tuple[MetricSpec, ...]:
    specs = []
    for name in target_names:
        specs.append(
            MetricSpec(
                name=f"auc_{name}",
                higher_is_better=True,
                description=f"Area Under ROC Curve for {name}",
            )
        )
        specs.append(
            MetricSpec(
                name=f"gauc_{name}",
                higher_is_better=True,
                description=f"Group Area Under ROC Curve for {name}",
            )
        )
    return tuple(specs)


__all__ = ["RankingMetricsConfig", "build_auc_metric_specs"]
