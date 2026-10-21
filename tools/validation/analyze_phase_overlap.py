#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.validation import perfetto_trace as trace_parser


def interval(event: dict[str, Any]) -> tuple[int, int]:
    start = int(event["ts_ns"])
    return start, start + int(event["dur_ns"])


def overlap_ms(a: tuple[int, int], b: tuple[int, int]) -> float:
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / 1e6


def hidden_before_deadline_ms(candidate: tuple[int, int], deadline_ns: int) -> float:
    return max(0, min(candidate[1], deadline_ns) - candidate[0]) / 1e6


def stat(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument(
        "--steps", required=True, help="comma-separated train_core steps"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    events = trace_parser.parse_trace(Path(args.trace))
    selected_steps = {int(item) for item in args.steps.split(",") if item}

    by_key: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["name"] != "user_callable.execute" or int(event["dur_ns"]) <= 0:
            continue
        step = event["payload"].get("step")
        function_id = event["payload"].get("function_id")
        if not isinstance(step, int) or not isinstance(function_id, str):
            raise TypeError("user callable trace lacks integer step or function id")
        fn = function_id.rsplit(".", 1)[-1]
        spmd = trace_parser.spmd_name(event["track"])
        rank = int(trace_parser.rank_name(event["track"]))
        by_key[(spmd, fn, rank, step)].append(event)

    rows: list[dict[str, Any]] = []
    for (spmd, fn, rank, step), train_events in sorted(by_key.items()):
        if spmd != "Dense" or fn != "train_core" or step not in selected_steps:
            continue
        train = max(train_events, key=lambda event: int(event["dur_ns"]))
        train_iv = interval(train)
        train_ms = int(train["dur_ns"]) / 1e6
        candidates = {
            "input_process_next": ("Input", "process", rank, step + 1),
            "dense_materialize_next": (
                "Dense",
                "prepare_sparse_materialize",
                rank,
                step + 1,
            ),
            "dense_bind_next": ("Dense", "prepare_jax_bind", rank, step + 1),
            "dense_scatter_prev": ("Dense", "scatter_sparse_grads", rank, step - 1),
            "dense_scatter_same": ("Dense", "scatter_sparse_grads", rank, step),
            "dense_materialize_same": (
                "Dense",
                "prepare_sparse_materialize",
                rank,
                step,
            ),
            "dense_bind_same": ("Dense", "prepare_jax_bind", rank, step),
        }
        overlaps = {}
        for name, key in candidates.items():
            candidate_events = by_key.get(key, [])
            if not candidate_events:
                overlaps[name] = {
                    "present": False,
                    "duration_ms": 0.0,
                    "overlap_ms": 0.0,
                    "hidden_ms": 0.0,
                    "hidden_fraction_of_candidate": 0.0,
                    "candidate_fraction_of_train": 0.0,
                }
                continue
            candidate = max(candidate_events, key=lambda event: int(event["dur_ns"]))
            candidate_ms = int(candidate["dur_ns"]) / 1e6
            candidate_iv = interval(candidate)
            ov = overlap_ms(train_iv, candidate_iv)
            hidden = hidden_before_deadline_ms(candidate_iv, train_iv[1])
            overlaps[name] = {
                "present": True,
                "duration_ms": candidate_ms,
                "overlap_ms": ov,
                "hidden_ms": hidden,
                "hidden_fraction_of_candidate": hidden / candidate_ms
                if candidate_ms
                else 0.0,
                "candidate_fraction_of_train": candidate_ms / train_ms
                if train_ms
                else 0.0,
                "overlap_fraction_of_train": ov / train_ms if train_ms else 0.0,
                "starts_before_train_end": int(candidate["ts_ns"]) < train_iv[1],
                "starts_before_train_start": int(candidate["ts_ns"]) < train_iv[0],
            }
        rows.append(
            {
                "rank": rank,
                "step": step,
                "train_core_ms": train_ms,
                "overlaps": overlaps,
            }
        )

    if not rows:
        raise ValueError("No selected Dense.train_core rows found")

    summary = {}
    for name in (
        "input_process_next",
        "dense_materialize_next",
        "dense_bind_next",
        "dense_scatter_prev",
        "dense_scatter_same",
        "dense_materialize_same",
        "dense_bind_same",
    ):
        present_rows = [
            row["overlaps"][name] for row in rows if row["overlaps"][name]["present"]
        ]
        summary[name] = {
            "present_count": len(present_rows),
            "positive_overlap_count": sum(
                1 for row in present_rows if row["overlap_ms"] > 0
            ),
            "duration_ms": stat([row["duration_ms"] for row in present_rows]),
            "overlap_ms": stat([row["overlap_ms"] for row in present_rows]),
            "hidden_ms": stat([row["hidden_ms"] for row in present_rows]),
            "hidden_fraction_of_candidate": stat(
                [row["hidden_fraction_of_candidate"] for row in present_rows]
            ),
            "candidate_fraction_of_train": stat(
                [row["candidate_fraction_of_train"] for row in present_rows]
            ),
            "started_before_train_end_count": sum(
                1 for row in present_rows if row["starts_before_train_end"]
            ),
        }

    payload = {
        "trace": str(Path(args.trace).resolve()),
        "selected_steps": sorted(selected_steps),
        "row_count": len(rows),
        "summary": summary,
        "rows": rows,
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
