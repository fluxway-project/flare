from __future__ import annotations

from dataclasses import dataclass

from fluxway.common.operator_policy import SchedulePolicy

RANKING_PHASE_ORDER = (
    "prepare_sparse_materialize",
    "prepare_jax_bind",
    "train_core",
    "scatter_sparse_grads",
    "update",
    "materialize_train_metrics",
)
RANKING_PHASE_DEPENDENCIES = (
    ("prepare_sparse_materialize", "prepare_jax_bind", 0),
    ("prepare_jax_bind", "train_core", 0),
    ("train_core", "scatter_sparse_grads", 0),
    ("train_core", "update", 0),
    ("train_core", "materialize_train_metrics", 0),
    ("update", "train_core", 1),
)
RANKING_LANE_ORDER = ("compute", "data")
RANKING_LANE_PRIORITIES = {"compute": 1000, "data": 100}
RANKING_PHASE_LANES = {
    "prepare_sparse_materialize": "data",
    "prepare_jax_bind": "data",
    "train_core": "compute",
    "scatter_sparse_grads": "data",
    "update": "compute",
    "materialize_train_metrics": "data",
}
RANKING_PHASE_PRIORITIES = {
    "train_core": 500,
    "prepare_sparse_materialize": 300,
    "prepare_jax_bind": 290,
    "scatter_sparse_grads": 20,
    "update": 480,
    "materialize_train_metrics": 10,
}


@dataclass(frozen=True)
class RankingScheduleContract:
    phase_order: tuple[str, ...] = RANKING_PHASE_ORDER
    phase_dependencies: tuple[tuple[str, str, int], ...] = RANKING_PHASE_DEPENDENCIES
    lane_order: tuple[str, ...] = RANKING_LANE_ORDER
    lane_priorities: dict[str, int] | None = None
    phase_lanes: dict[str, str] | None = None
    phase_priorities: dict[str, int] | None = None


def ranking_schedule_contract() -> RankingScheduleContract:
    return RankingScheduleContract(
        lane_priorities=dict(RANKING_LANE_PRIORITIES),
        phase_lanes=dict(RANKING_PHASE_LANES),
        phase_priorities=dict(RANKING_PHASE_PRIORITIES),
    )


def ranking_phase_schedule(phase_name: str) -> SchedulePolicy:
    if phase_name not in RANKING_PHASE_LANES:
        raise ValueError(
            f"unknown ranking phase {phase_name!r}; valid={sorted(RANKING_PHASE_LANES)}"
        )
    return SchedulePolicy.phase_graph(
        phase=phase_name,
        phase_order=RANKING_PHASE_ORDER,
        dependencies=RANKING_PHASE_DEPENDENCIES,
        lane=RANKING_PHASE_LANES[phase_name],
        lane_order=RANKING_LANE_ORDER,
        lane_priorities=RANKING_LANE_PRIORITIES,
        phase_priorities=RANKING_PHASE_PRIORITIES,
    )


__all__ = [
    "RANKING_LANE_ORDER",
    "RANKING_LANE_PRIORITIES",
    "RANKING_PHASE_DEPENDENCIES",
    "RANKING_PHASE_LANES",
    "RANKING_PHASE_ORDER",
    "RANKING_PHASE_PRIORITIES",
    "RankingScheduleContract",
    "ranking_phase_schedule",
    "ranking_schedule_contract",
]
