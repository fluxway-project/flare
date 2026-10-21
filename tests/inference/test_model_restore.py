from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from flare.inference.weights import source


class _StateDefinition:
    def __init__(self) -> None:
        self.validated: object | None = None

    def abstract_model_state(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {"weight": jax.ShapeDtypeStruct((2,), jnp.float32)}

    def validate_state(self, state: object) -> None:
        self.validated = state


def _cpu_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices("cpu")[:1]), ("dp",))


def test_model_restore_passes_target_layout_to_orbax_without_device_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, object]] = []

    class _Checkpointer:
        def __init__(self, handler: object) -> None:
            assert isinstance(handler, source.ocp.StandardCheckpointHandler)

        def restore(self, path: Path, *, args: object) -> object:
            calls.append((path, args))
            return args.item  # type: ignore[attr-defined]

        def close(self) -> None:
            return None

    monkeypatch.setattr(source.ocp, "Checkpointer", _Checkpointer)
    checkpoint_path = tmp_path / "model"
    checkpoint_path.mkdir()
    evidence = source.PublishedAssetEvidence.capture(
        checkpoint_path, checkpoint_path, None
    )

    def fail_device_put(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "model restore must not copy a complete tree after restore"
        )

    monkeypatch.setattr(source.jax, "device_put", fail_device_put)
    definition = _StateDefinition()
    mesh = _cpu_mesh()
    restored = source.OrbaxModelWeightAdapter(checkpoint_path, evidence).load(
        definition, mesh
    )

    assert restored is definition.validated
    assert len(calls) == 1
    restore_args = calls[0][1]
    target = restore_args.item["weight"]  # type: ignore[attr-defined]
    assert target.sharding == NamedSharding(mesh, P())


@pytest.mark.parametrize("mesh", [None, object()])
def test_model_restore_requires_explicit_mesh(tmp_path: Path, mesh: object) -> None:
    checkpoint_path = tmp_path / "model"
    checkpoint_path.mkdir()
    evidence = source.PublishedAssetEvidence.capture(
        checkpoint_path, checkpoint_path, None
    )
    with pytest.raises(TypeError, match="jax.sharding.Mesh"):
        source.OrbaxModelWeightAdapter(checkpoint_path, evidence).load(
            _StateDefinition(),
            mesh,  # type: ignore[arg-type]
        )
