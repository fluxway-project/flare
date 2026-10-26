from __future__ import annotations

import hashlib
from pathlib import Path

import msgpack

from tools.validation.inspect_checkpoint_source_cut import inspect


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(msgpack.packb(value, use_bin_type=True))


def test_inspector_checks_manifest_control_and_reader_cuts(tmp_path: Path) -> None:
    step = 5
    reader_workers = 2
    checkpoint_id = "checkpoint-5"
    source_partitions = [
        {"record": {"payload": msgpack.packb({"position": step}, use_bin_type=True)}}
        for _ in range(reader_workers)
    ]
    manifest = {
        "checkpoint_id": checkpoint_id,
        "cut": {"position": step},
        "framework_state": {
            "components": {
                "ingestion-planning": {"cut_position": step},
                "source-control": {"committed_cut": step},
            }
        },
        "spmds": {
            "Input": {"runtime_state": {"drain_boundary": {"committed_cut": step}}},
            "Reader": {
                "source_topology": {"world_size": reader_workers},
                "runtime_state": {"drain_boundary": {"committed_cut": step}},
                "components": {"source": {"partitions": source_partitions}},
            },
        },
    }
    manifest_path = tmp_path / "manifests" / checkpoint_id / "manifest.msgpack"
    _dump(manifest_path, manifest)
    _dump(
        tmp_path / "transactions" / f"{checkpoint_id}.msgpack",
        {
            "status": "committed",
            "manifest_digest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    )
    ref_path = tmp_path / "refs" / "steps" / str(step)
    ref_path.parent.mkdir(parents=True)
    ref_path.write_text(checkpoint_id)

    result = inspect(tmp_path, step, reader_workers)

    assert result["step"] == step
    assert result["reader_workers"] == reader_workers
    assert result["reader_positions"] == [step, step]
