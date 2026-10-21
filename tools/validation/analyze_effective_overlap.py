from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CRITICAL_RELATIONS = ("dense_materialize_next", "dense_scatter_prev")


def stat(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "total": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "total": float(sum(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p90": float(ordered[int((len(ordered) - 1) * 0.9)]),
        "max": float(ordered[-1]),
    }


def summarize_relation(rows: list[dict], relation: str) -> dict:
    present = [
        row["overlaps"][relation]
        for row in rows
        if row["overlaps"].get(relation, {}).get("present")
    ]
    durations = [float(item["duration_ms"]) for item in present]
    overlaps = [float(item["overlap_ms"]) for item in present]
    hidden_ms = [float(item["hidden_ms"]) for item in present]
    hidden = [float(item["hidden_fraction_of_candidate"]) for item in present]
    exposed = [
        float(item["duration_ms"]) - float(item["hidden_ms"]) for item in present
    ]
    starts_before_train_end = sum(
        bool(item.get("starts_before_train_end")) for item in present
    )
    positive_overlap = sum(value > 0 for value in overlaps)
    positive_hidden = sum(value > 0 for value in hidden_ms)
    total_duration = sum(durations)
    total_hidden = sum(hidden_ms)
    return {
        "present_count": len(present),
        "starts_before_train_end_count": starts_before_train_end,
        "positive_overlap_count": positive_overlap,
        "positive_hidden_count": positive_hidden,
        "duration_ms": stat(durations),
        "overlap_ms": stat(overlaps),
        "hidden_ms": stat(hidden_ms),
        "exposed_ms": stat(exposed),
        "hidden_fraction_of_candidate": stat(hidden),
        "total_hidden_fraction": total_hidden / total_duration
        if total_duration > 0
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-hidden-fraction", type=float, default=0.5)
    parser.add_argument("--min-positive-count", type=int, default=1)
    args = parser.parse_args()

    data = json.loads(args.overlap.read_text())
    rows = data["rows"]
    relation_summary = {
        relation: summarize_relation(rows, relation) for relation in CRITICAL_RELATIONS
    }
    train_core_ms = [float(row["train_core_ms"]) for row in rows]
    effective_data_overlap_ms = [
        sum(
            float(row["overlaps"][relation]["overlap_ms"])
            for relation in CRITICAL_RELATIONS
            if row["overlaps"].get(relation, {}).get("present")
        )
        for row in rows
    ]
    effective_data_overlap_fraction_of_train = [
        overlap / train if train > 0 else 0.0
        for overlap, train in zip(effective_data_overlap_ms, train_core_ms, strict=True)
    ]
    effective_data_hidden_ms = [
        sum(
            float(row["overlaps"][relation]["hidden_ms"])
            for relation in CRITICAL_RELATIONS
            if row["overlaps"].get(relation, {}).get("present")
        )
        for row in rows
    ]
    effective_data_hidden_fraction_of_train = [
        hidden / train if train > 0 else 0.0
        for hidden, train in zip(effective_data_hidden_ms, train_core_ms, strict=True)
    ]
    blockers = {}
    for relation, summary in relation_summary.items():
        relation_blockers = []
        if summary["positive_hidden_count"] < args.min_positive_count:
            relation_blockers.append("no consistently hidden work")
        if summary["total_hidden_fraction"] < args.min_hidden_fraction:
            relation_blockers.append("low hidden fraction")
        if summary["exposed_ms"].get("mean", 0.0) > summary["hidden_ms"].get(
            "mean", 0.0
        ):
            relation_blockers.append("more exposed than hidden")
        blockers[relation] = relation_blockers
    passed = all(not values for values in blockers.values())
    result = {
        "passed": passed,
        "overlap": str(args.overlap),
        "row_count": len(rows),
        "selected_steps": data.get("selected_steps"),
        "train_core_ms": stat(train_core_ms),
        "effective_data_overlap_ms": stat(effective_data_overlap_ms),
        "effective_data_overlap_fraction_of_train": stat(
            effective_data_overlap_fraction_of_train
        ),
        "effective_data_hidden_ms": stat(effective_data_hidden_ms),
        "effective_data_hidden_fraction_of_train": stat(
            effective_data_hidden_fraction_of_train
        ),
        "relations": relation_summary,
        "blockers": blockers,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
