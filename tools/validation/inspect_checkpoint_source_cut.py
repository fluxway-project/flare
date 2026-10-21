"""Inspect one committed Fluxway checkpoint and its durable source cut.

This is an offline consistency check. It does not claim crash/restart
exactly-once, which requires an actual interruption and reconciliation run.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import msgpack


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be a mapping")
    return value


def _load_manifest(root: Path, step: int) -> tuple[str, dict[str, Any]]:
    ref_path = root / "refs" / "steps" / str(step)
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)
    checkpoint_id = ref_path.read_text().strip()
    if not checkpoint_id:
        raise AssertionError(f"empty checkpoint ref: {ref_path}")
    manifest_path = root / "manifests" / checkpoint_id / "manifest.msgpack"
    manifest = _mapping(
        msgpack.unpackb(manifest_path.read_bytes(), raw=False),
        f"manifest {manifest_path}",
    )
    transaction_path = root / "transactions" / f"{checkpoint_id}.msgpack"
    transaction = _mapping(
        msgpack.unpackb(transaction_path.read_bytes(), raw=False),
        f"transaction {transaction_path}",
    )
    if transaction.get("status") != "committed":
        raise AssertionError(
            f"checkpoint transaction is not committed: {transaction.get('status')!r}"
        )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if transaction.get("manifest_digest") != digest:
        raise AssertionError("checkpoint manifest digest does not match transaction")
    if manifest.get("checkpoint_id") != checkpoint_id:
        raise AssertionError("checkpoint ref and manifest checkpoint IDs differ")
    return checkpoint_id, manifest


def inspect(
    root: Path, step: int, expected_reader_workers: int
) -> dict[str, Any]:
    checkpoint_id, manifest = _load_manifest(root, step)
    cut = _mapping(manifest.get("cut"), "manifest cut")
    if cut.get("position") != step:
        raise AssertionError(
            f"manifest cut differs from requested step: {cut.get('position')!r}"
        )
    framework = _mapping(manifest.get("framework_state"), "framework_state")
    framework_components = _mapping(
        framework.get("components"), "framework_state.components"
    )
    planning = _mapping(
        framework_components.get("ingestion-planning"), "ingestion-planning"
    )
    if planning.get("cut_position") != step:
        raise AssertionError("ingestion planning cut differs from checkpoint step")
    source_control = _mapping(
        framework_components.get("source-control"), "source-control"
    )
    if source_control.get("committed_cut") != step:
        raise AssertionError("source-control committed cut differs from checkpoint step")

    spmds = _mapping(manifest.get("spmds"), "manifest spmds")
    bad_spmds = []
    for spmd_id, state in spmds.items():
        runtime = _mapping(_mapping(state, f"spmd {spmd_id}").get("runtime_state"), f"spmd {spmd_id} runtime_state")
        drain = _mapping(runtime.get("drain_boundary"), f"spmd {spmd_id} drain_boundary")
        if drain.get("committed_cut") != step:
            bad_spmds.append(spmd_id)
    if bad_spmds:
        raise AssertionError(f"SPMD committed cuts differ from checkpoint step: {bad_spmds}")

    reader = _mapping(spmds.get("Reader"), "Reader SPMD")
    source_topology = _mapping(reader.get("source_topology"), "Reader source_topology")
    if source_topology.get("world_size") != expected_reader_workers:
        raise AssertionError(
            "Reader checkpoint topology differs: "
            f"expected={expected_reader_workers} actual={source_topology.get('world_size')!r}"
        )
    reader_components = _mapping(reader.get("components"), "Reader components")
    source = _mapping(reader_components.get("source"), "Reader source component")
    partitions = source.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        raise AssertionError("Reader source component has no partition checkpoints")
    if len(partitions) != expected_reader_workers:
        raise AssertionError(
            "Reader source partition count differs: "
            f"expected={expected_reader_workers} actual={len(partitions)}"
        )
    positions: list[int] = []
    for partition in partitions:
        record = _mapping(_mapping(partition, "Reader partition").get("record"), "Reader partition record")
        payload = record.get("payload")
        if not isinstance(payload, bytes):
            raise AssertionError("Reader partition checkpoint payload must be bytes")
        decoded = _mapping(msgpack.unpackb(payload, raw=False), "Reader partition payload")
        position = decoded.get("position")
        if not isinstance(position, int) or position != step:
            raise AssertionError(
                f"Reader partition position differs from checkpoint step: "
                f"expected={step} actual={position!r}"
            )
        positions.append(position)

    return {
        "checkpoint_id": checkpoint_id,
        "step": step,
        "reader_workers": expected_reader_workers,
        "reader_partitions": len(partitions),
        "reader_positions": sorted(positions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("step", type=int)
    parser.add_argument("--expected-reader-workers", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step < 0:
        raise ValueError("step must be non-negative")
    if args.expected_reader_workers <= 0:
        raise ValueError("expected reader worker count must be positive")
    result = inspect(args.checkpoint_dir, args.step, args.expected_reader_workers)
    print("checkpoint_source_cut_consistent=true")
    for key, value in result.items():
        print(f"checkpoint_{key}={value}")


if __name__ == "__main__":
    main()
