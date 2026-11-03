"""Benchmark ranking AUC/GAUC computation with GPU-free synthetic batches.

This measures the Ray Data metric window only, not training throughput or GPU
collection. Alternating the parallelism order limits simple cold-start bias.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import ray
from ray.data import DataContext

from flare.jax.runtime_env import build_jax_runtime_env
from flare.runtime.fluxway.resources import normalize_resource_vector
from recipes.recsys.ranking.metrics.ray_data import RayDataMetricsComputer

_GIB = 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--rows-per-batch", type=int, default=20_000)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--group-cardinality", type=int, default=20_000)
    parser.add_argument("--parallelisms", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--memory-gib", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    counts = (
        args.batches,
        args.rows_per_batch,
        args.targets,
        args.group_cardinality,
        args.rounds,
        args.memory_gib,
        *args.parallelisms,
    )
    if any(value <= 0 for value in counts):
        parser.error("batch, target, resource, and repeat counts must be positive")

    runtime_env = build_jax_runtime_env(
        xla_cache_dir=str(Path("tmp/ranking-metric-xla").resolve()),
        platform="cpu",
    )
    ray.init(address=args.ray_address, runtime_env=runtime_env, logging_level="ERROR")
    try:
        DataContext.get_current().enable_progress_bars = False
        names = tuple(f"target_{index}" for index in range(args.targets))
        rng = np.random.default_rng(args.seed)

        def computer(parallelism: int) -> RayDataMetricsComputer:
            return RayDataMetricsComputer(
                target_names=names,
                negative_factor=None,
                ray_component_limits={
                    "CPU": parallelism,
                    "memory": args.memory_gib * _GIB,
                },
                resource_normalizer=normalize_resource_vector,
            )

        uploader = computer(args.parallelisms[0])
        refs = tuple(
            uploader.put_batch(
                scores=rng.random(
                    (args.rows_per_batch, args.targets), dtype=np.float32
                ),
                labels=rng.integers(
                    0, 2, (args.rows_per_batch, args.targets), dtype=np.int8
                ),
                groups=rng.integers(
                    0, args.group_cardinality, args.rows_per_batch, dtype=np.int64
                ),
            )
            for _ in range(args.batches)
        )
        timings: dict[int, list[float]] = {value: [] for value in args.parallelisms}
        expected_samples = args.rows_per_batch * args.batches
        for round_index in range(args.rounds):
            order = (
                args.parallelisms
                if round_index % 2 == 0
                else list(reversed(args.parallelisms))
            )
            for parallelism in order:
                started = time.perf_counter()
                result = computer(parallelism).compute(refs)
                elapsed = time.perf_counter() - started
                if any(
                    int(result[name]["sample_count"]) != expected_samples
                    for name in names
                ):
                    raise RuntimeError("metric sample count changed between runs")
                timings[parallelism].append(elapsed)
                print(
                    f"metric_ray_parallelism={parallelism} seconds={elapsed:.3f} "
                    f"samples={expected_samples}",
                    flush=True,
                )
        summary = {
            "samples": expected_samples,
            "targets": args.targets,
            "group_cardinality": args.group_cardinality,
            "batches": args.batches,
            "rounds": args.rounds,
            "results": {
                str(parallelism): {
                    "seconds": durations,
                    "median_seconds": statistics.median(durations),
                }
                for parallelism, durations in timings.items()
            },
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
