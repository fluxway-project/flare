"""Reconcile successful Reader ranges; this does not prove crash exactly-once."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditPhase:
    path: Path
    workers: int
    start_step: int
    end_step: int

    def __post_init__(self) -> None:
        if self.workers <= 0 or not 0 <= self.start_step < self.end_step:
            raise ValueError("audit phase requires workers > 0 and start < end")


def _integer(record: dict[str, Any], name: str, minimum: int = 0) -> int:
    value = record[name]
    if type(value) is not int or value < minimum:
        raise ValueError(f"audit {name} must be an integer >= {minimum}")
    return value


def reconcile(
    phases: tuple[AuditPhase, ...],
    *,
    batch_size: int,
    source_rows: int,
    require_exhaustion: bool = False,
) -> dict[str, Any]:
    if not phases or batch_size <= 0 or source_rows <= 0:
        raise ValueError("phases, positive batch_size and source_rows are required")
    identities: set[tuple[str, str]] = set()
    summaries = []
    expected_step = phases[0].start_step
    terminal_step = (source_rows + batch_size - 1) // batch_size
    for phase in phases:
        if phase.start_step != expected_step or phase.end_step > terminal_step:
            raise ValueError("audit phases must be contiguous within the source")
        by_step: dict[int, list[tuple[int, int]]] = defaultdict(list)
        record_count = 0
        for rank in range(phase.workers):
            path = phase.path.with_name(
                f"{phase.path.stem}_rank_{rank}{phase.path.suffix}"
            )
            payload = json.loads(path.read_text())
            if _integer(payload, "rank") != rank:
                raise ValueError(f"audit file rank mismatch: {path}")
            records = payload["records"]
            if not isinstance(records, list):
                raise TypeError(f"audit records must be a list: {path}")
            for record in records:
                if _integer(record, "rank") != rank:
                    raise ValueError(f"audit record rank mismatch: {path}")
                step = _integer(record, "step")
                if not phase.start_step <= step < phase.end_step:
                    raise ValueError(f"audit step outside phase: {step}")
                if _integer(record, "global_batch_size", 1) != batch_size:
                    raise ValueError("audit global batch size mismatch")
                start = _integer(record, "global_start")
                count = _integer(record, "cardinality", 1)
                logical_start = _integer(record, "logical_start")
                if _integer(record, "logical_end", 1) - logical_start != count:
                    raise ValueError("audit logical range cardinality mismatch")
                identity = (record["source_identity"], record["schema_digest"])
                if any(not isinstance(value, str) or not value for value in identity):
                    raise ValueError("audit source and schema identities are required")
                identities.add(identity)
                if len(identities) != 1:
                    raise ValueError("audit source or schema changed between reads")
                by_step[step].append((start, start + count))
                record_count += 1
        for step in range(phase.start_step, phase.end_step):
            cursor = step * batch_size
            for start, end in sorted(by_step.pop(step, ())):
                if start != cursor:
                    raise ValueError(
                        f"audit gap or overlap at step {step}: {cursor}, {start}"
                    )
                cursor = end
            if cursor != (step + 1) * batch_size:
                raise ValueError(f"audit incomplete batch at step {step}")
        first = phase.start_step * batch_size
        last = phase.end_step * batch_size
        valid = min(last, source_rows) - min(first, source_rows)
        summaries.append(
            {
                "start_step": phase.start_step,
                "end_step": phase.end_step,
                "reader_workers": phase.workers,
                "records": record_count,
                "valid_rows": valid,
                "padding_rows": last - first - valid,
            }
        )
        expected_step = phase.end_step
    if require_exhaustion and (
        phases[0].start_step != 0 or expected_step != terminal_step
    ):
        raise ValueError("audit does not cover the complete source")
    return {
        "evidence": "successful_assigned_read_ranges",
        "crash_exactly_once_proven": False,
        "input_receipt_proven": False,
        "source_rows": source_rows,
        "batch_size": batch_size,
        "source_exhaustion_covered": phases[0].start_step == 0
        and expected_step == terminal_step,
        "valid_rows": sum(phase["valid_rows"] for phase in summaries),
        "padding_rows": sum(phase["padding_rows"] for phase in summaries),
        "phases": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        action="append",
        nargs=4,
        required=True,
        metavar=("AUDIT_PATH", "WORKERS", "START_STEP", "END_STEP"),
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--source-rows", type=int, required=True)
    parser.add_argument("--require-exhaustion", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phases = tuple(
        AuditPhase(Path(path), int(workers), int(start), int(end))
        for path, workers, start, end in args.phase
    )
    result = reconcile(
        phases,
        batch_size=args.batch_size,
        source_rows=args.source_rows,
        require_exhaustion=args.require_exhaustion,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
