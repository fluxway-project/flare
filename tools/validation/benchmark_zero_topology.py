"""Compare global versus node-local ZeRO ownership with exact global DP gradients.

This measures communication and an Adam-like vector update, not model forward,
data I/O, or full train_core. It requires a Ray cluster with one GPU per actor.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def build_macro_update(mesh, *, mode: str, world_size: int, local_size: int):
    """Build one exact-global-DP update with global or per-node ZeRO owners."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    if mode == "global":
        dp_axes, zero_axis, zero_size = "dp", "dp", world_size
        grad_spec = P("dp", None)
    elif mode == "node_local":
        dp_axes, zero_axis, zero_size = ("node", "zero"), "zero", local_size
        grad_spec = P(("node", "zero"), None)
    else:
        raise ValueError(f"unknown mode: {mode}")

    def update(model, first_moment, second_moment, accumulated_gradient):
        global_gradient = (
            jax.lax.pmean(accumulated_gradient[0].astype(jnp.bfloat16), dp_axes).astype(
                jnp.float32
            )
            / 3.0
        )
        shard_index = jax.lax.axis_index(zero_axis)
        local_gradient = global_gradient.reshape(zero_size, -1)[shard_index]
        local_model = model.reshape(zero_size, -1)[shard_index]
        next_first = 0.9 * first_moment + 0.1 * local_gradient
        next_second = 0.999 * second_moment + 0.001 * local_gradient**2
        next_model = local_model - 0.001 * next_first / (jnp.sqrt(next_second) + 1e-8)
        replicated_model = jax.lax.all_gather(next_model, zero_axis, axis=0, tiled=True)
        return replicated_model, next_first, next_second

    mapped = jax.shard_map(
        update,
        mesh=mesh,
        in_specs=(P(), P(zero_axis), P(zero_axis), grad_spec),
        out_specs=(P(), P(zero_axis), P(zero_axis)),
        check_vma=False,
    )
    return jax.jit(mapped, donate_argnums=(0, 1, 2))


def self_test():
    """Check that two nodes of local ZeRO match one global ZeRO group."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = np.asarray(jax.devices()[:4], dtype=object)
    if len(devices) != 4:
        raise RuntimeError("self_test requires four JAX devices")
    outputs = {}
    for mode in ("global", "node_local"):
        mesh = (
            Mesh(devices, ("dp",))
            if mode == "global"
            else Mesh(devices.reshape(2, 2), ("node", "zero"))
        )
        zero_axis = "dp" if mode == "global" else "zero"
        grad_spec = P("dp", None) if mode == "global" else P(("node", "zero"), None)
        model = jax.device_put(
            1.0 + jnp.arange(16, dtype=jnp.float32) / 16,
            NamedSharding(mesh, P()),
        )
        first = jax.device_put(
            jnp.zeros(16, dtype=jnp.float32), NamedSharding(mesh, P(zero_axis))
        )
        second = jax.device_put(
            jnp.zeros(16, dtype=jnp.float32), NamedSharding(mesh, P(zero_axis))
        )
        grad = jax.device_put(
            jnp.arange(1, 5, dtype=jnp.float32)[:, None]
            * (1 + jnp.arange(16, dtype=jnp.float32)[None, :] / 16)
            * 3,
            NamedSharding(mesh, grad_spec),
        )
        update = build_macro_update(mesh, mode=mode, world_size=4, local_size=2)
        outputs[mode] = np.asarray(update(model, first, second, grad)[0])
    np.testing.assert_allclose(outputs["global"], outputs["node_local"], rtol=0, atol=0)
    print("ZERO_TOPOLOGY_SELF_TEST=PASS")


@ray.remote(num_cpus=1, num_gpus=1, max_restarts=0)
class Rank:
    def gpu_id(self):
        (gpu,) = ray.get_gpu_ids()
        return int(gpu)

    def free_port(self):
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", 0))
            return sock.getsockname()[1]

    def run(
        self,
        rank: int,
        world_size: int,
        local_size: int,
        coordinator: str,
        mode: str,
        elements: int,
        warmup: int,
        iterations: int,
    ):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.experimental import multihost_utils
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=world_size,
            process_id=rank,
            local_device_ids=[0],
        )
        devices = np.asarray(jax.devices(), dtype=object)
        if mode == "global":
            mesh = Mesh(devices, ("dp",))
            grad_spec, zero_axis, zero_size = P("dp", None), "dp", world_size
        else:
            mesh = Mesh(
                devices.reshape(world_size // local_size, local_size), ("node", "zero")
            )
            grad_spec, zero_axis, zero_size = (
                P(("node", "zero"), None),
                "zero",
                local_size,
            )
        device = jax.local_devices()[0]

        def global_from_local(value, shape, spec):
            return jax.make_array_from_single_device_arrays(
                shape, NamedSharding(mesh, spec), [jax.device_put(value, device)]
            )

        model = global_from_local(jnp.ones((elements,), jnp.float32), (elements,), P())
        moment_shape = (elements // zero_size,)
        first = global_from_local(
            jnp.zeros(moment_shape, jnp.float32), (elements,), P(zero_axis)
        )
        second = global_from_local(
            jnp.zeros(moment_shape, jnp.float32), (elements,), P(zero_axis)
        )
        rank_gradient = global_from_local(
            jnp.full((1, elements), rank + 1, jnp.float32),
            (world_size, elements),
            grad_spec,
        )
        add = jax.jit(lambda total, current: total + current, donate_argnums=(0,))
        update = build_macro_update(
            mesh, mode=mode, world_size=world_size, local_size=local_size
        )
        multihost_utils.sync_global_devices("zero_topology_initialized")

        def macro_step():
            total = jnp.zeros_like(rank_gradient)
            for _ in range(3):
                total = add(total, rank_gradient)
            jax.block_until_ready(total)
            update_started = time.perf_counter()
            nonlocal model, first, second
            model, first, second = update(model, first, second, total)
            jax.block_until_ready((model, first, second))
            return time.perf_counter() - update_started

        for _ in range(warmup):
            macro_step()
        multihost_utils.sync_global_devices("zero_topology_measure_start")
        updates = []
        started = time.perf_counter()
        for _ in range(iterations):
            updates.append(macro_step())
        multihost_utils.sync_global_devices("zero_topology_measure_end")
        elapsed = time.perf_counter() - started
        head = np.asarray(model[:8]).tolist()
        return {
            "rank": rank,
            "elapsed_seconds": elapsed,
            "update_seconds": updates,
            "model_head": head,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("global", "node_local"), required=True)
    parser.add_argument("--elements", type=int, default=67_108_864)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--skip-gpu-topology-validation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    world_size = args.nodes * args.gpus_per_node
    if args.elements <= 0 or args.elements % world_size:
        raise ValueError("elements must be positive and divisible by all GPUs")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    ray.init(address="auto")
    nodes = sorted(
        [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) >= args.gpus_per_node
        ],
        key=lambda node: node["NodeManagerAddress"],
    )[: args.nodes]
    if len(nodes) != args.nodes:
        raise RuntimeError("insufficient GPU nodes")
    env = {
        "JAX_PLATFORMS": "cuda,cpu",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_CLIENT_MEM_FRACTION": "0.90",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
    }
    if args.skip_gpu_topology_validation:
        env["XLA_PJRT_GPU_VALIDATE_TOPOLOGY"] = "0"
    actors_by_node = []
    for node in nodes:
        actors = [
            Rank.options(
                runtime_env={"env_vars": env},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node["NodeID"], soft=False
                ),
            ).remote()
            for _ in range(args.gpus_per_node)
        ]
        assigned = ray.get([actor.gpu_id.remote() for actor in actors])
        if sorted(assigned) != list(range(args.gpus_per_node)):
            raise RuntimeError(f"unexpected GPU assignment: {assigned}")
        actors_by_node.extend(
            actor for _, actor in sorted(zip(assigned, actors, strict=True))
        )
    try:
        coordinator = f"{nodes[0]['NodeManagerAddress']}:{ray.get(actors_by_node[0].free_port.remote())}"
        results = sorted(
            ray.get(
                [
                    actor.run.remote(
                        rank,
                        world_size,
                        args.gpus_per_node,
                        coordinator,
                        args.mode,
                        args.elements,
                        args.warmup,
                        args.iterations,
                    )
                    for rank, actor in enumerate(actors_by_node)
                ]
            ),
            key=lambda item: item["rank"],
        )
        elapsed = max(result["elapsed_seconds"] for result in results)
        update_wall = [
            max(result["update_seconds"][index] for result in results)
            for index in range(args.iterations)
        ]
        output = {
            "mode": args.mode,
            "world_size": world_size,
            "gpus_per_node": args.gpus_per_node,
            "elements": args.elements,
            "gradient_bytes_per_rank": args.elements * 4,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "elapsed_seconds": elapsed,
            "macro_steps_per_second": args.iterations / elapsed,
            "median_update_seconds": statistics.median(update_wall),
            "update_wall_seconds": update_wall,
            "model_heads": [result["model_head"] for result in results],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(
            {key: value for key, value in output.items() if key != "model_heads"},
            flush=True,
        )
    finally:
        for actor in actors_by_node:
            ray.kill(actor, no_restart=True)


if __name__ == "__main__":
    main()
