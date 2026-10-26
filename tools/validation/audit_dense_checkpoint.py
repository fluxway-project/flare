"""Restore a committed dense checkpoint on CPU through its public owners."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np
from fluxway.common.types.checkpoint import (
    CheckpointRuntime,
    CheckpointRuntimeRole,
    RestoreSession,
    StepReference,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer import (
    CheckpointRepository,
    OrbaxCheckpointComponent,
)
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from flare.jax.training.nnx.state import NnxTrainingStateDefinition
from flare.runtime.diagnostics.state_audit import fingerprint_state


def file_inventory(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


async def audit(
    *,
    checkpoint_root: Path,
    step: int,
    expected_optimizer_step: int,
    definition: NnxTrainingStateDefinition,
    spmd_id: str,
    component_id: str,
    scratch: Path,
) -> dict[str, Any]:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("offline dense audit requires JAX_PLATFORMS=cpu")
    if os.environ.get("SYSTEM_ENV") or os.environ.get("GRAVITINO_TOKEN"):
        raise ValueError("offline dense audit must not receive source credentials")
    if step <= 0 or expected_optimizer_step <= 0:
        raise ValueError("source and optimizer steps must be positive")
    if scratch.is_relative_to(checkpoint_root):
        raise ValueError("audit scratch must be outside the protected checkpoint")
    if not isinstance(definition, NnxTrainingStateDefinition):
        raise TypeError("state factory must return NnxTrainingStateDefinition")
    before = file_inventory(checkpoint_root)
    manifest = CheckpointRepository(checkpoint_root).resolve(StepReference(step))
    if manifest.cut.position != step:
        raise ValueError("resolved checkpoint cut differs from requested step")
    saved = manifest.spmds[spmd_id]
    scratch.mkdir(parents=True, exist_ok=False)
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1], dtype=object), ("audit",))
    layout = definition.checkpoint_layout(NamedSharding(mesh, PartitionSpec()))
    owner = SimpleNamespace(state=None)

    def refuse_save(instance, session):
        raise RuntimeError("offline dense audit cannot save a checkpoint")

    def accept_state(instance, state, session):
        definition.validate_checkpoint_payload(state, context="offline dense audit")
        instance.state = state

    component = OrbaxCheckpointComponent(
        component_id, refuse_save, accept_state, lambda instance, session: layout
    )
    session = RestoreSession(
        instance=owner,
        checkpoint_id=manifest.checkpoint_id,
        cut=manifest.cut,
        spmd_id=spmd_id,
        rank=0,
        saved_topology=saved.source_topology,
        restore_topology=TopologySnapshot(
            world_size=1, mesh_axis_names=("audit",), mesh_axis_shapes=(1,)
        ),
        component_id=component_id,
        component_checkpoint=saved.components[component_id],
    )
    await component.open(
        CheckpointRuntime(
            role=CheckpointRuntimeRole.WORKER,
            repository_root=str(scratch),
            spmd_id=spmd_id,
            rank=0,
        )
    )
    try:
        await component.restore(session)
    finally:
        await component.close()
    if file_inventory(checkpoint_root) != before:
        raise RuntimeError("protected checkpoint file inventory changed during audit")
    state = owner.state
    update_count = definition.validate_checkpoint_payload(state)
    if update_count != expected_optimizer_step:
        raise ValueError(
            f"optimizer update count mismatch: expected={expected_optimizer_step} actual={update_count}"
        )
    leaves = jtu.tree_leaves(state["optimizer_state"]["step"])
    if len(leaves) != 1 or int(np.asarray(leaves[0])) != update_count:
        raise ValueError("NNX optimizer step differs from checkpoint update count")
    fingerprints = {
        name: fingerprint_state(state[key])
        for name, key in (("model", "model_state"), ("optimizer", "optimizer_state"))
    }
    if any(value["nan_count"] or value["inf_count"] for value in fingerprints.values()):
        raise ValueError("committed dense state contains NaN or Inf")
    return {
        "checkpoint_id": manifest.checkpoint_id,
        "source_step": step,
        "saved_world_size": saved.source_topology.world_size,
        "restore_world_size": 1,
        "restore_platform": "cpu",
        "optimizer_update_count": update_count,
        "optimizer_scalar_integer_leaves": {
            jtu.keystr(path): int(np.asarray(leaf))
            for path, leaf in jtu.tree_flatten_with_path(state["optimizer_state"])[0]
            if leaf.shape == () and np.issubdtype(leaf.dtype, np.integer)
        },
        "fingerprints": fingerprints,
        "protected_checkpoint_files_unchanged": True,
        "sparse_state_verified": False,
        "crash_exactly_once_proven": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--expected-optimizer-step", type=int, required=True)
    parser.add_argument("--state-definition-factory", required=True)
    parser.add_argument("--factory-kwargs", default="{}")
    parser.add_argument("--spmd-id", required=True)
    parser.add_argument("--component-id", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, scratch, output = (
        p.resolve() for p in (args.checkpoint_root, args.scratch, args.output)
    )
    if output.is_relative_to(root) or output.exists():
        raise ValueError("audit output must be new and outside the checkpoint")
    module, name = args.state_definition_factory.split(":", 1)
    kwargs = json.loads(args.factory_kwargs)
    if not isinstance(kwargs, dict):
        raise TypeError("state factory arguments must be a JSON object")
    definition = getattr(importlib.import_module(module), name)(**kwargs)
    result = asyncio.run(
        audit(
            checkpoint_root=root,
            step=args.step,
            expected_optimizer_step=args.expected_optimizer_step,
            definition=definition,
            spmd_id=args.spmd_id,
            component_id=args.component_id,
            scratch=scratch,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as destination:
        destination.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
