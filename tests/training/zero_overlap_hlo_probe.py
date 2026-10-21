from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from bf16_distributed_runtime_probe import (
    CheckpointFactory,
    CheckpointModel,
    create_runtime,
)
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.jax.training.precision import PrecisionPolicy
from flare.training.state import TrainingStateLifecycle
from flare.training.zero import ZeroStage


def forward(
    module: CheckpointModel, inputs: Mapping[str, jax.Array]
) -> dict[str, jax.Array]:
    module.stat[...] = module.stat[...] + jnp.mean(inputs["x"])
    return {"logits": inputs["x"] @ module.weight.get_value() + module.bias.get_value()}


def loss(
    predictions: Mapping[str, jax.Array], targets: Mapping[str, jax.Array]
) -> jax.Array:
    return jnp.mean((predictions["logits"] - targets["labels"]) ** 2)


for stage in (ZeroStage.OPTIMIZER_STATE, ZeroStage.GRADIENT, ZeroStage.PARAMETER):
    runtime = create_runtime(
        definition=NnxTrainingStateDefinition.from_factory(
            CheckpointFactory(), precision_policy=PrecisionPolicy.bf16()
        ),
        stage=stage,
        world_size=2,
        lifecycle=TrainingStateLifecycle.FRESH_INITIALIZE,
        forward_fn=forward,
        loss_fn=loss,
    )
    runtime._build_functions(("x",))
    input_sharding = NamedSharding(runtime.mesh, P("dp"))
    x = jax.device_put(jnp.ones((2, 5), dtype=jnp.float32), input_sharding)
    labels = jax.device_put(jnp.zeros((2, 3), dtype=jnp.float32), input_sharding)
    compiled = runtime.micro_fn_jit.lower(
        runtime.flat_state,
        (x,),
        (labels, jnp.asarray(2, dtype=jnp.int32)),
    ).compile()
    hlo = compiled.as_text()
    lines = hlo.splitlines()
    print(
        f"STAGE={stage.value} PLATFORM={jax.default_backend()} HLO_LINES={len(lines)}"
    )
    for index, line in enumerate(lines):
        if any(
            token in line
            for token in (
                "all-reduce",
                "reduce-scatter",
                "async-start",
                "async-done",
                " dot(",
                " dot-general(",
            )
        ):
            print(f"{index + 1}: {line[:350]}")
    entry = hlo.split("ENTRY ", 1)[1]
    entry_lines = entry.splitlines()
    for index, line in enumerate(entry_lines):
        if "reduce-scatter-start =" in line:
            done_index = next(
                next_index
                for next_index in range(index + 1, len(entry_lines))
                if "reduce-scatter-done =" in entry_lines[next_index]
            )
            print(f"START_DONE_GAP={done_index - index - 1}")
            for between in entry_lines[index + 1 : done_index]:
                print(f"BETWEEN: {between[:250]}")
