import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_audit", type=Path)
    parser.add_argument("restored_audit", type=Path)
    parser.add_argument("--initial-world-size", type=int, required=True)
    parser.add_argument("--restored-world-size", type=int, required=True)
    parser.add_argument("--initial-steps", type=int, required=True)
    parser.add_argument("--restored-steps", type=int, required=True)
    args = parser.parse_args()
    for name in (
        "initial_world_size",
        "restored_world_size",
        "initial_steps",
        "restored_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    return args


def observed(path: Path, expected_world_size: int) -> dict[int, int]:
    observations: dict[int, int] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        event = json.loads(line)
        if set(event) != {"position", "rank", "world_size"}:
            raise AssertionError(f"invalid Reader audit event at {path}:{line_number}")
        position = int(event["position"])
        rank = int(event["rank"])
        world_size = int(event["world_size"])
        if world_size != expected_world_size:
            raise AssertionError(
                f"Reader world size differs at {path}:{line_number}: "
                f"expected={expected_world_size} actual={world_size}"
            )
        if position in observations:
            raise AssertionError(f"duplicate Reader position in {path}: {position}")
        observations[position] = rank
    return observations


def main() -> None:
    args = parse_args()
    expected_initial = {
        step: step % args.initial_world_size for step in range(args.initial_steps)
    }
    restored_start = args.initial_steps
    expected_restored = {
        step: step % args.restored_world_size
        for step in range(
            restored_start,
            restored_start + args.restored_steps,
        )
    }
    actual_initial = observed(args.initial_audit, args.initial_world_size)
    actual_restored = observed(args.restored_audit, args.restored_world_size)
    if actual_initial != expected_initial:
        raise AssertionError(
            "initial Reader interleave differs: "
            f"expected={expected_initial} actual={actual_initial}"
        )
    if actual_restored != expected_restored:
        raise AssertionError(
            "restored Reader interleave differs: "
            f"expected={expected_restored} actual={actual_restored}"
        )
    expected = {**expected_initial, **expected_restored}
    print(
        f"interleave_topology={args.initial_world_size}->"
        f"{args.restored_world_size} positions=0..{max(expected)} exact=true"
    )


if __name__ == "__main__":
    main()
