"""Partition one host's real resources across isolated Ray test nodes."""

from __future__ import annotations

import ipaddress
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict, cast

import psutil
import ray
from ray.actor import ActorHandle
from ray.cluster_utils import Cluster
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


class NodeResources(TypedDict):
    node_id: str
    cpus: int
    gpu_uuids: tuple[str, ...]
    worker_memory_bytes: int
    object_store_memory_bytes: int


def visible_gpu_uuids() -> tuple[str, ...]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    )
    devices = tuple(
        tuple(value.strip() for value in line.split(",")) for line in text.splitlines()
    )
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selected is None:
        return tuple(uuid for _, uuid in devices)
    result = []
    for token in selected.split(","):
        matches = [
            uuid
            for index, uuid in devices
            if token == index or uuid.startswith(token) and token
        ]
        if len(matches) != 1 or matches[0] in result:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES does not identify distinct available GPUs"
            )
        result.append(matches[0])
    return tuple(result)


@contextmanager
def local_ray_cluster(
    *,
    nodes: int,
    node_ip_address: str,
    temp_dir: Path,
    inventory_path: Path,
    object_store_bytes_per_node: int = 1024**3,
    control_reserve_bytes: int = 4 * 1024**3,
):
    if ipaddress.ip_address(node_ip_address).is_loopback:
        raise ValueError("Ray test nodes require an explicit non-loopback host address")
    gpu_uuids = visible_gpu_uuids()
    cpu_count = len(os.sched_getaffinity(0))
    if nodes < 2 or len(gpu_uuids) < nodes or cpu_count < nodes:
        raise ValueError("simulated nodes require distinct real GPUs and CPUs")
    if object_store_bytes_per_node <= 0 or control_reserve_bytes <= 0:
        raise ValueError("object store and control reserve must be positive")
    available_memory = psutil.virtual_memory().available
    memory = (
        available_memory - nodes * object_store_bytes_per_node - control_reserve_bytes
    )
    if memory <= 0:
        raise ValueError("host memory is insufficient for the configured cluster")
    cluster = Cluster()
    records: list[NodeResources] = []
    try:
        for index in range(nodes):
            assigned_gpus = gpu_uuids[index::nodes]
            cpus = cpu_count // nodes + (index < cpu_count % nodes)
            node_memory = memory // nodes
            node = cluster.add_node(
                num_cpus=cpus,
                num_gpus=len(assigned_gpus),
                memory=node_memory,
                object_store_memory=object_store_bytes_per_node,
                include_dashboard=False,
                temp_dir=str(temp_dir),
                node_ip_address=node_ip_address,
                env_vars={"CUDA_VISIBLE_DEVICES": ",".join(assigned_gpus)},
            )
            records.append(
                {
                    "node_id": node.node_id,
                    "cpus": cpus,
                    "gpu_uuids": assigned_gpus,
                    "worker_memory_bytes": node_memory,
                    "object_store_memory_bytes": object_store_bytes_per_node,
                }
            )
        inventory_path.write_text(
            json.dumps(
                {
                    "scope": "multiple_ray_nodes_on_one_physical_host",
                    "address": cluster.address,
                    "host_available_memory_bytes": available_memory,
                    "control_reserve_bytes": control_reserve_bytes,
                    "nodes": records,
                },
                indent=2,
            )
            + "\n"
        )
        ray.init(
            address=cluster.address,
            _node_ip_address=node_ip_address,
            log_to_driver=False,
        )

        class GPUProbe:
            def inspect(self):
                import jax
                import jax.numpy as jnp

                if len(jax.devices("gpu")) != 1:
                    raise RuntimeError("GPU probe must see exactly its assigned device")
                return {
                    "node_id": ray.get_runtime_context().get_node_id(),
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "sum": int(jnp.arange(64).sum().block_until_ready()),
                }

        remote_gpu_probe = ray.remote(num_cpus=0, num_gpus=1)(GPUProbe)
        probes: list[ActorHandle] = []
        try:
            for record in records:
                for _ in record["gpu_uuids"]:
                    probes.append(
                        cast(
                            ActorHandle,
                            remote_gpu_probe.options(
                                scheduling_strategy=NodeAffinitySchedulingStrategy(
                                    record["node_id"], soft=False
                                ),
                                runtime_env={
                                    "env_vars": {
                                        "JAX_PLATFORMS": "cuda,cpu",
                                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                                    }
                                },
                            ).remote(),
                        )
                    )
            results = ray.get([probe.inspect.remote() for probe in probes])
            expected = {uuid for record in records for uuid in record["gpu_uuids"]}
            if {result["cuda_visible_devices"] for result in results} != expected:
                raise ValueError(
                    "Ray GPU visibility does not preserve disjoint physical UUIDs"
                )
            if any(result["sum"] != 2016 for result in results):
                raise ValueError("Ray GPU arithmetic probe failed")
            inventory = json.loads(inventory_path.read_text())
            inventory["gpu_probes"] = results
            inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
        finally:
            for probe in probes:
                ray.kill(probe, no_restart=True)
        yield cluster.address
    finally:
        ray.shutdown()
        cluster.shutdown()
