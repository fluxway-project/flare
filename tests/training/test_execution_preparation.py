from __future__ import annotations

from typing import Any

import jax

from flare.jax.training.engines.distributed import initialize_distributed_jax
from flare.training.execution import (
    DistributedTrainStepExecutor,
    GradientAccumulationPolicy,
    MacroUpdatePolicy,
    TrainingExecutionPlan,
    ZeroExecutionPolicy,
)
from flare.training.zero import ZeroStage


def test_distributed_jax_initialization_uses_ray_process_local_device(
    monkeypatch,
) -> None:
    """Ray workers see one GPU; Fluxway local_rank is not a device ordinal."""
    initialized: list[dict[str, Any]] = []
    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        jax.distributed,
        "initialize",
        lambda **kwargs: initialized.append(kwargs),
    )
    monkeypatch.setattr(jax, "process_index", lambda: 2)
    monkeypatch.setattr(jax, "process_count", lambda: 4)

    initialize_distributed_jax(
        rank=2,
        world_size=4,
        local_rank=2,
        local_world_size=4,
        rank0_address="127.0.0.1:12345",
    )

    assert initialized == [
        {
            "coordinator_address": "127.0.0.1:12345",
            "num_processes": 4,
            "process_id": 2,
            "local_device_ids": [0],
        }
    ]


def test_distributed_executor_exposes_side_effect_free_preparation_seam() -> None:
    prepared: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def prepare(inputs: dict[str, Any], targets: dict[str, Any]) -> None:
        prepared.append((inputs, targets))

    executor = DistributedTrainStepExecutor(
        plan=TrainingExecutionPlan(
            zero=ZeroExecutionPolicy(ZeroStage.OPTIMIZER_STATE),
            accumulation=GradientAccumulationPolicy(1),
            update=MacroUpdatePolicy(),
        ),
        stage=ZeroStage.OPTIMIZER_STATE,
        prepare_execution=prepare,
        micro_step=lambda _inputs, _targets: None,
        micro_step_counter=lambda: 0,
        macro_update=lambda _window: False,
        checkpoint_finalize=lambda: False,
        checkpoint_capture=dict,
        checkpoint_layout=dict,
        checkpoint_restore=lambda _state: None,
        audit_state_capture=dict,
        module_fingerprint_capture=dict,
    )

    inputs = {"batch": object()}
    targets = {"labels": object()}
    executor.prepare_execution(inputs, targets)

    assert prepared == [(inputs, targets)]


def test_operator_phase_events_include_monotonic_sequence() -> None:
    from recipes.recsys.ranking.workers.train_worker import RecsysBackboneTrainSPMD

    class _Tracer:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def record_event(self, **event: Any) -> None:
            self.events.append(event)

    worker = object.__new__(RecsysBackboneTrainSPMD)
    worker.execution_tracer = _Tracer()
    worker._execution_event_sequence = 0
    worker.micro_step = 0
    worker.rank = 1

    worker._record_execution_phase(
        "operator_prepare_begin", payload={"phase": "compile"}
    )
    worker._record_execution_phase("operator_prepare_end", payload={"phase": "compile"})

    assert [event["stage"] for event in worker.execution_tracer.events] == [
        "operator_prepare_begin",
        "operator_prepare_end",
    ]
    assert [
        event["payload"]["sequence"] for event in worker.execution_tracer.events
    ] == [
        0,
        1,
    ]
    assert all(
        event["spmd_id"] == "RecsysBackboneTrain"
        for event in worker.execution_tracer.events
    )
