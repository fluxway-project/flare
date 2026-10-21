from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax import nnx

from flare.inference.weights import bind_inference_weights
from flare.jax.layers.state import NnxModelDefinition
from flare.mesh import build_1d_device_mesh
from flare.runtime.fluxway.transport import make_transport_tensor, transport_payload
from recipes.recsys.ranking.workers import train_worker
from recipes.recsys.ranking.workers.train_worker import RecsysBackboneEvalSPMD


class _Model(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.asarray([1.5], dtype=jnp.float32))

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"logits": inputs["side_info"] * self.weight}


@pytest.fixture
def make_eval_worker(tmp_path: Path) -> Callable[..., RecsysBackboneEvalSPMD]:
    definition = NnxModelDefinition.from_initializer(_Model)
    asset = tmp_path / "weights/steps/7"
    (asset / "sparse").mkdir(parents=True)
    with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as checkpointer:
        checkpointer.save(
            asset / "model",
            args=ocp.args.StandardSave(nnx.state(definition.initialize())),
        )
    weights = bind_inference_weights(
        asset,
        definition,
        build_1d_device_mesh(1, axis_name="dp"),
        partition_mappings={},
    )

    def create(forward_fn: Callable[..., Any]) -> RecsysBackboneEvalSPMD:
        worker = object.__new__(RecsysBackboneEvalSPMD)
        worker.rank = 0
        worker.world_size = 1
        worker.__init__(
            model_definition=definition,
            sparse_groups=(),
            side_info_groups=("side_info",),
            RankingOutput=SimpleNamespace,
            sparse_group_widths={},
            sparse_tensor_backend="jax",
            forward_fn=forward_fn,
            weight_source=weights,
            sparse_embedding_dtype="float32",
        )
        return worker

    return create


def _batch(values: tuple[float, ...]) -> dict[str, Any]:
    return {
        "side_info": make_transport_tensor(
            np.asarray(values, dtype=np.float32)[:, None]
        ),
        "labels": make_transport_tensor(np.zeros(len(values), dtype=np.int32)),
        "samples_valid_count": make_transport_tensor(
            np.asarray([len(values)], dtype=np.int32)
        ),
    }


def test_eval_reuses_compiled_forward_with_explicit_inputs_and_weights(
    make_eval_worker: Callable[..., RecsysBackboneEvalSPMD],
) -> None:
    traces = []

    def forward(module: _Model, inputs: dict[str, Any]) -> dict[str, Any]:
        traces.append(inputs["side_info"].shape)
        return module(inputs)

    worker = make_eval_worker(forward)
    for values in ((0.0, 1.0), (1.0, 2.0)):
        result = worker.step(**_batch(values))
        expected = jax.nn.sigmoid(jnp.asarray(values)[:, None] * 1.5)
        np.testing.assert_allclose(
            transport_payload(result.scores), expected, rtol=1e-6
        )
    worker.module.weight[...] = jnp.asarray([2.5], dtype=jnp.float32)
    updated = worker.step(**_batch((1.0, 2.0)))
    np.testing.assert_allclose(
        transport_payload(updated.scores),
        jax.nn.sigmoid(jnp.asarray([[1.0], [2.0]]) * 2.5),
        rtol=1e-6,
    )
    assert traces == [(2, 1)]
    assert transport_payload(updated.scores).dtype == np.float32
    np.testing.assert_array_equal(transport_payload(updated.labels), [0, 0])


@pytest.mark.parametrize("output", [{}, (jnp.ones((2, 1)),)])
def test_eval_releases_input_lease_when_forward_output_is_invalid(
    make_eval_worker: Callable[..., RecsysBackboneEvalSPMD],
    monkeypatch: pytest.MonkeyPatch,
    output: Any,
) -> None:
    released = []
    lease = SimpleNamespace(release=lambda: released.append(True))
    monkeypatch.setattr(
        train_worker,
        "to_jax_array_with_lease",
        lambda value: (jnp.asarray(value), lease),
    )
    worker = make_eval_worker(lambda _module, _inputs: output)

    with pytest.raises(TypeError, match="mapping with 'logits'"):
        worker.step(**_batch((0.0, 1.0)))

    assert released == [True]
