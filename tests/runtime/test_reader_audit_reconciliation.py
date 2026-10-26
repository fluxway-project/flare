from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.reconcile_reader_audit import AuditPhase, reconcile


def write_phase(root: Path, name: str, start: int, end: int) -> AuditPhase:
    path = root / f"{name}.json"
    for rank in range(2):
        records = [
            {
                "rank": rank,
                "step": step,
                "global_start": step * 8 + rank * 4,
                "cardinality": 4,
                "logical_start": step * 8 + rank * 4,
                "logical_end": step * 8 + rank * 4 + 4,
                "global_batch_size": 8,
                "source_identity": "source",
                "schema_digest": "schema",
            }
            for step in range(start, end)
        ]
        path.with_name(f"{name}_rank_{rank}.json").write_text(
            json.dumps({"rank": rank, "records": records})
        )
    return AuditPhase(path, 2, start, end)


def test_reconciles_phases_and_partial_terminal_batch(tmp_path):
    phases = (
        write_phase(tmp_path, "first", 0, 1),
        write_phase(tmp_path, "second", 1, 3),
    )
    result = reconcile(phases, batch_size=8, source_rows=21, require_exhaustion=True)
    assert result["valid_rows"] == 21
    assert result["padding_rows"] == 3
    assert result["source_exhaustion_covered"] is True
    assert result["crash_exactly_once_proven"] is False
    assert result["input_receipt_proven"] is False


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("global_start", 1, "gap or overlap"),
        ("global_start", 0, "gap or overlap"),
        ("cardinality", 3, "cardinality mismatch"),
        ("step", 3, "outside phase"),
        ("rank", 0, "rank mismatch"),
        ("schema_digest", "other", "schema changed"),
        ("source_identity", "other", "source or schema changed"),
        ("global_batch_size", 4, "batch size mismatch"),
        ("global_start", True, "must be an integer"),
    ],
)
def test_rejects_corrupt_read_coverage(tmp_path, field, value, match):
    phase = write_phase(tmp_path, "reader", 0, 3)
    path = tmp_path / "reader_rank_1.json"
    payload = json.loads(path.read_text())
    payload["records"][0][field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        reconcile((phase,), batch_size=8, source_rows=21, require_exhaustion=True)


def test_rejects_missing_batch_and_source_truncation(tmp_path):
    phase = write_phase(tmp_path, "reader", 0, 2)
    with pytest.raises(ValueError, match="complete source"):
        reconcile((phase,), batch_size=8, source_rows=21, require_exhaustion=True)
    path = tmp_path / "reader_rank_1.json"
    payload = json.loads(path.read_text())
    payload["records"].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete batch"):
        reconcile((phase,), batch_size=8, source_rows=21)
