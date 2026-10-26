"""Validate compiled Reader lanes against attested runtime node placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate(payload: dict[str, Any], *, minimum_nodes: int = 2) -> dict[str, Any]:
    report = payload["allocation"]
    if report["status"] != "READY":
        raise ValueError("locality validation requires a READY allocation")
    attested = report["attested_node_ids"]
    readers = attested[payload["reader_spmd_id"]]
    inputs = attested[payload["input_spmd_id"]]
    groups = payload["reader_execution_groups"]
    if len(set(inputs)) < minimum_nodes:
        raise ValueError("Input workers do not span the required runtime nodes")
    if len(groups) < 2 or any(len(group) != len(inputs) for group in groups):
        raise ValueError("compiled layout must contain multiple Readers per Input lane")
    flattened = [rank for group in groups for rank in group]
    if sorted(flattened) != list(range(len(readers))):
        raise ValueError("compiled Reader groups must partition physical ranks")
    planned = {
        (item["spmd_id"], item["rank"]): item["node_id"]
        for item in report["assignments"]
    }
    for spmd_id in (payload["reader_spmd_id"], payload["input_spmd_id"]):
        for rank, node in enumerate(attested[spmd_id]):
            if planned[(spmd_id, rank)] != node:
                raise ValueError("planned worker node differs from runtime attestation")
    lanes = []
    for lane, target_node in enumerate(inputs):
        ranks = [group[lane] for group in groups]
        if any(readers[rank] != target_node for rank in ranks):
            raise ValueError(f"Reader lane {lane} crosses runtime nodes")
        lanes.append(
            {"input_rank": lane, "reader_ranks": ranks, "node_id": target_node}
        )
    return {
        "evidence": "compiled_reader_lanes_and_attested_runtime_placement",
        "runtime_nodes": len(set(inputs)),
        "lanes": lanes,
        "measured_transport_bytes": False,
        "physical_multi_host_proven": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--minimum-nodes", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        json.loads(args.audit.read_text()), minimum_nodes=args.minimum_nodes
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
