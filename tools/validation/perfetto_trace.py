#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as proto


def annotation_value(annotation: Any) -> int | str | bool | float | None:
    field = annotation.WhichOneof("value")
    if field is None:
        return None
    return getattr(annotation, field)


def full_track_name(
    uuid: int,
    track_names: dict[int, str],
    track_parents: dict[int, int],
) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current = uuid
    while current and current not in seen:
        seen.add(current)
        name = track_names.get(current)
        if name:
            parts.append(name)
        current = track_parents.get(current, 0)
    return "/".join(reversed(parts)) if parts else str(uuid)


def parse_trace(path: Path) -> list[dict[str, Any]]:
    trace = proto.Trace()
    trace.ParseFromString(path.read_bytes())
    track_names: dict[int, str] = {}
    track_parents: dict[int, int] = {}
    packets = list(trace.packet)
    for packet in packets:
        if packet.HasField("track_descriptor"):
            desc = packet.track_descriptor
            track_names[int(desc.uuid)] = desc.name
            track_parents[int(desc.uuid)] = int(desc.parent_uuid)

    events: list[dict[str, Any]] = []
    stacks: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)

    for packet in packets:
        if packet.HasField("track_descriptor"):
            continue
        if not packet.HasField("track_event"):
            continue
        event = packet.track_event
        track_uuid = int(event.track_uuid)
        name = event.name
        payload = {
            annotation.name: annotation_value(annotation)
            for annotation in event.debug_annotations
        }
        base = {
            "ts_ns": int(packet.timestamp),
            "track_uuid": track_uuid,
            "track": full_track_name(track_uuid, track_names, track_parents),
            "name": name,
            "payload": payload,
        }
        if event.type == proto.TrackEvent.TYPE_SLICE_BEGIN:
            stacks[(track_uuid, name)].append(base)
        elif event.type == proto.TrackEvent.TYPE_SLICE_END:
            stack = stacks[(track_uuid, name)]
            if not stack:
                raise RuntimeError(
                    f"slice end without begin: {name} track={track_uuid}"
                )
            begin = stack.pop()
            duration = int(packet.timestamp) - int(begin["ts_ns"])
            if duration < 0:
                raise RuntimeError(f"negative duration for {name}: {duration}")
            events.append({**begin, "dur_ns": duration})
        else:
            events.append({**base, "dur_ns": 0})

    unclosed = sum(len(stack) for stack in stacks.values())
    if unclosed:
        raise RuntimeError(f"trace has {unclosed} unclosed slices")
    return events


def spmd_name(track: str) -> str:
    parts = track.split("/")
    if "SPMDs" in parts:
        idx = parts.index("SPMDs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError(f"trace track has no SPMD identity: {track!r}")


def rank_name(track: str) -> str:
    parts = track.split("/")
    for part in parts:
        if part.startswith("rank:"):
            return part.split(":", 1)[1]
    raise ValueError(f"trace track has no rank identity: {track!r}")


def stats(values: list[float], batch_size: int) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "max_ms": ordered[-1],
        "throughput_from_mean_samples_per_sec": batch_size
        / (statistics.fmean(values) / 1000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.global_batch_size <= 0:
        raise ValueError("--global-batch-size must be positive")

    events = parse_trace(Path(args.trace))
    event_counts = Counter(event["name"] for event in events)
    names = sorted(event_counts)
    step_events = [
        event
        for event in events
        if isinstance(event["payload"].get("step"), int)
        or isinstance(event["payload"].get("completed_count"), int)
    ]
    user_events = [
        event for event in events if event["name"] == "user_callable.execute"
    ]
    dense_user_events = [
        event for event in user_events if spmd_name(event["track"]) == "Dense"
    ]

    event_duration_by_name: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event["dur_ns"] > 0:
            event_duration_by_name[event["name"]].append(event["dur_ns"] / 1e6)

    by_user_op: dict[str, list[float]] = defaultdict(list)
    by_user_fn: dict[str, list[float]] = defaultdict(list)
    by_user_spmd_fn: dict[str, list[float]] = defaultdict(list)
    by_dense_step: dict[int, list[float]] = defaultdict(list)
    user_step_breakdown: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in user_events:
        dur_ms = event["dur_ns"] / 1e6
        step = event["payload"].get("step")
        key = f"{spmd_name(event['track'])}.{event['payload'].get('function_id')}"
        by_user_op[str(event["payload"].get("op_id"))].append(dur_ms)
        by_user_fn[str(event["payload"].get("function_id"))].append(dur_ms)
        by_user_spmd_fn[key].append(dur_ms)
        if isinstance(step, int):
            user_step_breakdown[key][step].append(dur_ms)
        if event in dense_user_events and isinstance(step, int):
            by_dense_step[step].append(dur_ms)

    selected_event_step_breakdown: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in events:
        step = event["payload"].get("step")
        if not isinstance(step, int) or event["dur_ns"] <= 0:
            continue
        if spmd_name(event["track"]) != "Dense":
            continue
        if event["name"] not in {
            "local_step.wait_inputs",
            "worker.assemble_input",
            "local_step.call_user",
            "local_step.send_output",
            "local_step.await_send",
            "local_step.commit",
            "local_step_plan.execute",
        }:
            continue
        selected_event_step_breakdown[event["name"]][step].append(event["dur_ns"] / 1e6)

    dense_step_max_ms = {
        step: max(values) for step, values in sorted(by_dense_step.items())
    }
    dense_step_span_ms: dict[int, float] = {}
    for step, values in sorted(by_dense_step.items()):
        step_dense = [
            event
            for event in dense_user_events
            if event["payload"].get("step") == step and event["dur_ns"] > 0
        ]
        if step_dense:
            start = min(event["ts_ns"] for event in step_dense)
            end = max(event["ts_ns"] + event["dur_ns"] for event in step_dense)
            dense_step_span_ms[step] = (end - start) / 1e6

    top_events_by_total_ms = sorted(
        (
            {
                "name": name,
                "count": len(values),
                "total_ms": sum(values),
                "mean_ms": statistics.fmean(values),
                "max_ms": max(values),
            }
            for name, values in event_duration_by_name.items()
        ),
        key=lambda row: row["total_ms"],
        reverse=True,
    )[:30]

    result = {
        "trace": args.trace,
        "packet_event_count": len(events),
        "event_name_count": len(names),
        "first_event_names": names[:40],
        "step_payload_min": min(
            [
                event["payload"]["step"]
                for event in step_events
                if "step" in event["payload"]
            ],
            default=None,
        ),
        "step_payload_max": max(
            [
                event["payload"]["step"]
                for event in step_events
                if "step" in event["payload"]
            ],
            default=None,
        ),
        "top_event_counts": event_counts.most_common(30),
        "top_events_by_total_ms": top_events_by_total_ms,
        "user_callable_by_op": {
            op: stats(values, args.global_batch_size)
            for op, values in sorted(by_user_op.items())
        },
        "user_callable_by_function": {
            fn: stats(values, args.global_batch_size)
            for fn, values in sorted(by_user_fn.items())
        },
        "user_callable_by_spmd_function": {
            key: stats(values, args.global_batch_size)
            for key, values in sorted(by_user_spmd_fn.items())
        },
        "user_callable_step_max_by_spmd_function": {
            key: {step: max(values) for step, values in sorted(step_values.items())}
            for key, step_values in sorted(user_step_breakdown.items())
        },
        "dense_event_step_max_ms": {
            name: {step: max(values) for step, values in sorted(step_values.items())}
            for name, step_values in sorted(selected_event_step_breakdown.items())
        },
        "dense_user_callable_step_max_ms": dense_step_max_ms,
        "dense_user_callable_step_max_stats": stats(
            list(dense_step_max_ms.values()), args.global_batch_size
        ),
        "dense_user_callable_step_span_ms": dense_step_span_ms,
        "dense_user_callable_step_span_stats": stats(
            list(dense_step_span_ms.values()), args.global_batch_size
        ),
        "completed_count_values": sorted(
            {
                event["payload"]["completed_count"]
                for event in step_events
                if isinstance(event["payload"].get("completed_count"), int)
            }
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
