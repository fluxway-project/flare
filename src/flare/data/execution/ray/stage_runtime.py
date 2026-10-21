from __future__ import annotations

import ray.data

from flare.data.execution.plan.physical import PhysicalPlan


def apply_physical_plan(ds, plan: PhysicalPlan):
    for stage in plan.stages:
        ds = ds.map_batches(
            stage.transform,
            fn_constructor_kwargs=stage.constructor_kwargs,
            batch_size=stage.batch_size,
            batch_format="pyarrow",
            num_cpus=stage.num_cpus,
            memory=stage.memory_bytes,
            compute=ray.data.ActorPoolStrategy(
                min_size=stage.min_actors,
                max_size=stage.max_actors,
                max_tasks_in_flight_per_actor=stage.tasks_in_flight,
            ),
            max_restarts=0,
        )
    return ds
