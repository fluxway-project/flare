from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from flare.data.execution.plan.physical import PhysicalPlan, PhysicalStage
from flare.data.execution.ray.stage_runtime import apply_physical_plan


class _First:
    pass


class _Second:
    pass


def _stage(stage_id: str, transform: type, memory: int) -> PhysicalStage:
    return PhysicalStage(
        stage_id=stage_id,
        transform=transform,
        constructor_kwargs={"stage": stage_id},
        min_actors=2,
        max_actors=4,
        tasks_in_flight=3,
        num_cpus=1.0,
        memory_bytes=memory,
        batch_size=1024,
    )


def test_runtime_applies_stages_in_declared_order() -> None:
    ds = MagicMock()
    ds.map_batches.return_value = ds
    plan = PhysicalPlan(
        (_stage("first", _First, 1024), _stage("second", _Second, 2048))
    )

    result = apply_physical_plan(ds, plan)

    assert result is ds
    calls = ds.map_batches.call_args_list
    assert [call.args[0] for call in calls] == [_First, _Second]
    assert [call.kwargs["memory"] for call in calls] == [1024, 2048]
    assert [call.kwargs["batch_size"] for call in calls] == [1024, 1024]
    assert all(call.kwargs["batch_format"] == "pyarrow" for call in calls)
    assert all(call.kwargs["max_restarts"] == 0 for call in calls)
    assert all(
        call.kwargs["compute"].max_tasks_in_flight_per_actor == 3 for call in calls
    )


def test_physical_plan_rejects_invalid_state() -> None:
    with pytest.raises(ValueError, match="contain stages"):
        PhysicalPlan(())
    with pytest.raises(ValueError, match="unique"):
        PhysicalPlan((_stage("same", _First, 1), _stage("same", _Second, 1)))
    with pytest.raises(ValueError, match="resources"):
        PhysicalStage("bad", _First, {}, 1, 1, 0, 1.0, 1, 1024)
