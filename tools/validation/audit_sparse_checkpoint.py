"""Read committed native sparse model and optimizer state through public owners."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from fluxway.common.types.checkpoint import (
    CheckpointRuntime,
    CheckpointRuntimeRole,
    RestoreSession,
    StepReference,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer import CheckpointRepository
from fluxway.integrations.sparse_embedding.checkpoint import (
    LocalLanceStorageFactory,
    SparseEmbeddingCheckpointComponent,
)

from flare.runtime.diagnostics.state_audit import fingerprint_state
from tools.validation.audit_dense_checkpoint import file_inventory


def summarize_store(store: Any, *, expected_optimizer_step: int) -> dict[str, Any]:
    if expected_optimizer_step <= 0 or not store.optimizer_enabled():
        raise ValueError(
            "sparse audit requires optimizer state and a positive step bound"
        )
    pages: dict[int, list[dict[str, np.ndarray]]] = {
        dim: [] for dim in store.table_dims()
    }
    optimizer_widths = {}
    for image in store.export_page_images():
        dim = image["embedding_dim"]
        keys = np.asarray(image["keys"], dtype=np.uint64).astype(store.key_dtype())
        steps = np.asarray(image["steps"], dtype=np.uint32)
        if keys.ndim != 1 or steps.shape != keys.shape:
            raise ValueError("native sparse per-page keys and steps must be aligned")
        count = keys.size
        if not count:
            continue
        optimizer = np.asarray(
            image["optimizer_states"], dtype=store.optimizer_dtype()
        ).reshape(count, -1)
        width = optimizer.shape[1]
        if dim in optimizer_widths and optimizer_widths[dim] != width:
            raise ValueError("native sparse optimizer width differs within a dimension")
        optimizer_widths[dim] = width
        pages[dim].append(
            {
                "keys": keys,
                "values": np.asarray(
                    image["values"], dtype=store.value_dtype()
                ).reshape(count, dim),
                "optimizer": optimizer,
                "steps": steps,
            }
        )
    states: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in ("keys", "values", "optimizer", "steps")
    }
    for dim, chunks in pages.items():
        if not chunks:
            continue
        combined = {
            name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
            for name in states
        }
        chunks.clear()
        keys = combined["keys"]
        if keys.ndim != 1 or np.unique(keys).size != keys.size:
            raise ValueError("native sparse export contains duplicate or invalid keys")
        steps = combined["steps"]
        if steps.shape != keys.shape or np.any(steps > expected_optimizer_step):
            raise ValueError(
                "native sparse per-key update count exceeds the expected bound"
            )
        order = keys.argsort()
        for name, value in combined.items():
            states[name][dim] = value[order]
        combined.clear()
    entries = sum(values.size for values in states["keys"].values())
    if entries != store.total_size():
        raise ValueError("native sparse export does not cover the restored store")
    fingerprints = (
        {name: fingerprint_state(tree) for name, tree in states.items()}
        if entries
        else {}
    )
    if any(value["nan_count"] or value["inf_count"] for value in fingerprints.values()):
        raise ValueError("committed sparse state contains NaN or Inf")
    steps = states["steps"].values()
    return {
        "entry_count": entries,
        "dimensions": list(pages),
        "dimension_slots": store.num_slots(),
        "optimizer_elements_per_key": optimizer_widths,
        "key_step_min": min(
            (int(value.min()) for value in steps if value.size), default=0
        ),
        "key_step_max": max(
            (int(value.max()) for value in states["steps"].values() if value.size),
            default=0,
        ),
        "fingerprints": fingerprints,
    }


async def audit(
    *,
    checkpoint_root: Path,
    step: int,
    expected_optimizer_step: int,
    store_factories: dict[str, tuple[str, Callable[[], Any]]],
) -> dict[str, Any]:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("offline sparse audit requires JAX_PLATFORMS=cpu")
    if os.environ.get("SYSTEM_ENV") or os.environ.get("GRAVITINO_TOKEN"):
        raise ValueError("offline sparse audit must not receive source credentials")
    if step <= 0 or expected_optimizer_step <= 0 or not store_factories:
        raise ValueError(
            "sparse audit requires positive source/update steps and stores"
        )
    before = file_inventory(checkpoint_root)
    manifest = CheckpointRepository(checkpoint_root).resolve(StepReference(step))
    if manifest.cut.position != step:
        raise ValueError("resolved checkpoint cut differs from requested step")
    results = {}
    for spmd_id, (component_id, create_store) in store_factories.items():
        store = create_store()
        saved = manifest.spmds[spmd_id]
        component = SparseEmbeddingCheckpointComponent(
            LocalLanceStorageFactory(), component_id=component_id
        )
        await component.open(
            CheckpointRuntime(
                role=CheckpointRuntimeRole.WORKER,
                repository_root=str(checkpoint_root),
                spmd_id=spmd_id,
                rank=0,
            )
        )
        try:
            await component.restore(
                RestoreSession(
                    instance=SimpleNamespace(store=store),
                    checkpoint_id=manifest.checkpoint_id,
                    cut=manifest.cut,
                    spmd_id=spmd_id,
                    rank=0,
                    saved_topology=saved.source_topology,
                    restore_topology=TopologySnapshot(1, ("audit",), (1,)),
                    component_id=component_id,
                    component_checkpoint=saved.components[component_id],
                )
            )
            results[spmd_id] = {
                "component_id": component_id,
                "saved_world_size": saved.source_topology.world_size,
                "restore_world_size": 1,
                **summarize_store(
                    store, expected_optimizer_step=expected_optimizer_step
                ),
            }
        finally:
            await component.close()
            del store
    if file_inventory(checkpoint_root) != before:
        raise RuntimeError(
            "protected checkpoint file inventory changed during sparse audit"
        )
    return {
        "checkpoint_id": manifest.checkpoint_id,
        "source_step": step,
        "expected_optimizer_update_bound": expected_optimizer_step,
        "components": results,
        "protected_checkpoint_files_unchanged": True,
        "reference_equivalence_proven": False,
        "crash_exactly_once_proven": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--expected-optimizer-step", type=int, required=True)
    parser.add_argument("--store-factory", required=True)
    parser.add_argument("--factory-kwargs", default="{}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = (p.resolve() for p in (args.checkpoint_root, args.output))
    if output.is_relative_to(root) or output.exists():
        raise ValueError("audit output must be new and outside the checkpoint")
    module, name = args.store_factory.split(":", 1)
    kwargs = json.loads(args.factory_kwargs)
    if not isinstance(kwargs, dict):
        raise TypeError("store factory arguments must be a JSON object")
    store_factories = getattr(importlib.import_module(module), name)(**kwargs)
    result = asyncio.run(
        audit(
            checkpoint_root=root,
            step=args.step,
            expected_optimizer_step=args.expected_optimizer_step,
            store_factories=store_factories,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as destination:
        destination.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
