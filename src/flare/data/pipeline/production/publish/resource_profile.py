"""flare.data.pipeline.production.publish.resource_profile — Hardware detection and topology planning.

Provides:
  - HardwareProfile: Frozen dataclass describing detected hardware capabilities.
  - HardwareProfiler: Runtime hardware detection (CPU, memory, GPU, library availability).
  - StageTopology: Frozen dataclass describing the per-stage actor layout for
    the stage-split pipeline (source → parse → encode → sink).
  - StageTopologyPlanner: Computes a StageTopology from cluster resources +
    backend choice. Shared by bench_throughput.py and the orchestrator so
    autoscaling and bench runs use the same rules.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass

import pyarrow as pa

logger = logging.getLogger(__name__)

DEFAULT_WRITE_TASK_NUM_CPUS = 1.0


@dataclass(frozen=True, slots=True)
class RayClusterCapacity:
    cpu_count: int
    heap_bytes: int
    object_store_bytes: int
    gpu_count: int

    @classmethod
    def from_mapping(cls, resources: Mapping[str, float]) -> RayClusterCapacity:
        required = {}
        for name in ("CPU", "memory", "object_store_memory"):
            if name not in resources:
                raise RuntimeError(f"Ray cluster resource {name!r} is unavailable")
            value = int(resources[name])
            if value <= 0:
                raise RuntimeError(
                    f"Ray cluster resource {name!r} must be positive, got {value}"
                )
            required[name] = value
        gpu_count = int(resources.get("GPU", 0))
        if gpu_count < 0:
            raise RuntimeError(f"Ray reported negative GPU capacity: {gpu_count}")
        return cls(
            cpu_count=required["CPU"],
            heap_bytes=required["memory"],
            object_store_bytes=required["object_store_memory"],
            gpu_count=gpu_count,
        )


def estimate_arrow_row_bytes(schema: pa.Schema) -> int:
    """Return a conservative encoded row-width estimate from schema metadata.

    Variable-length feature lists carry their maximum source width in the
    ``pad_length`` field metadata produced by ``FeatureProcessor``.  Refusing
    an unbounded list is deliberate: sizing a streaming block without a bound
    would turn an OOM into a probabilistic failure.
    """

    total = 0
    for field in schema:
        data_type = field.type
        if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
            metadata = field.metadata or {}
            raw_length = metadata.get(b"pad_length")
            if raw_length is None:
                raise ValueError(
                    f"cannot estimate row width for unbounded list field {field.name!r}"
                )
            length = int(raw_length)
            value_type = data_type.value_type
            if not hasattr(value_type, "bit_width") or value_type.bit_width <= 0:
                raise ValueError(
                    f"cannot estimate row width for list field {field.name!r}"
                )
            total += 8 + (length * value_type.bit_width // 8)
        elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
            total += 16
        elif hasattr(data_type, "bit_width") and data_type.bit_width > 0:
            total += data_type.bit_width // 8
        else:
            raise ValueError(f"cannot estimate row width for field {field.name!r}")
    if total <= 0:
        raise ValueError("output schema has no measurable row width")
    return total


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Detected hardware capabilities of the execution environment.

    Attributes:
        cpu_count: Number of available CPU cores.
        memory_bytes: Total system memory in bytes.
        gpu_count: Number of detected NVIDIA GPUs.
        gpu_memory_bytes: Per-GPU memory in bytes (0 if no GPU).
        has_nvcomp: Whether nvcomp/kvikio GPU compression library is importable.
    """

    cpu_count: int
    memory_bytes: int
    gpu_count: int
    gpu_memory_bytes: int
    has_nvcomp: bool

    @property
    def has_gpu(self) -> bool:
        """True if at least one GPU is available."""
        return self.gpu_count > 0

    @property
    def has_full_gpu_stack(self) -> bool:
        """True if GPU + nvcomp are all available."""
        return self.has_gpu and self.has_nvcomp


class HardwareProfiler:
    """Detects runtime hardware capabilities.

    Probes CPU count, memory, GPU count/memory, and availability of
    GPU-accelerated libraries (nvcomp, cudf).
    """

    def detect(self) -> HardwareProfile:
        """Perform hardware detection and return a frozen profile."""
        cpu_count = self._detect_cpu_count()
        memory_bytes = self._detect_memory()
        gpu_count, gpu_memory = self._detect_gpu()
        has_nvcomp = self._probe_import("kvikio")

        profile = HardwareProfile(
            cpu_count=cpu_count,
            memory_bytes=memory_bytes,
            gpu_count=gpu_count,
            gpu_memory_bytes=gpu_memory,
            has_nvcomp=has_nvcomp,
        )

        logger.info(
            "Hardware profile: CPUs=%d, Memory=%.1fGB, GPUs=%d, "
            "GPU_Memory=%.1fGB, nvcomp=%s",
            profile.cpu_count,
            profile.memory_bytes / (1024**3),
            profile.gpu_count,
            profile.gpu_memory_bytes / (1024**3),
            profile.has_nvcomp,
        )
        return profile

    def _detect_cpu_count(self) -> int:
        """Detect available CPU cores, respecting cgroup limits."""
        try:
            import ray
        except ImportError:
            ray = None
        if ray is not None and ray.is_initialized():
            value = int(ray.cluster_resources()["CPU"])
        else:
            detected = os.cpu_count()
            if detected is None:
                raise RuntimeError("operating system did not report CPU capacity")
            value = int(detected)
        if value <= 0:
            raise RuntimeError(f"detected non-positive CPU capacity: {value}")
        return value

    def _detect_memory(self) -> int:
        """Detect total system memory in bytes."""
        try:
            import psutil
        except ImportError:
            psutil = None
        if psutil is not None:
            value = int(psutil.virtual_memory().total)
        else:
            try:
                with open("/proc/meminfo", encoding="utf-8") as stream:
                    line = next(item for item in stream if item.startswith("MemTotal:"))
            except (FileNotFoundError, OSError, StopIteration) as exc:
                raise RuntimeError(
                    "memory capacity is unavailable; install psutil or provide "
                    "a platform memory source"
                ) from exc
            value = int(line.split()[1]) * 1024
        if value <= 0:
            raise RuntimeError(f"detected non-positive memory capacity: {value}")
        return value

    def _detect_gpu(self) -> tuple[int, int]:
        """Detect GPU count and per-GPU memory.

        Returns:
            (gpu_count, per_gpu_memory_bytes)
        """
        try:
            import ray
        except ImportError:
            ray = None
        if ray is not None and ray.is_initialized():
            count = int(ray.cluster_resources().get("GPU", 0))
            if count > 0:
                return count, self._nvidia_smi_memory()
            return 0, 0

        if shutil.which("nvidia-smi") is None:
            return 0, 0
        count = self._nvidia_smi_count()
        if count > 0:
            return count, self._nvidia_smi_memory()

        return 0, 0

    def _nvidia_smi_count(self) -> int:
        """Get GPU count via nvidia-smi."""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 6 and "No devices" in result.stdout:
            return 0
        if result.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi GPU count probe failed with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("nvidia-smi returned no GPU count")
        return len(lines)

    def _nvidia_smi_memory(self) -> int:
        """Get per-GPU memory in bytes via nvidia-smi."""
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi memory probe failed with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not values:
            raise RuntimeError("nvidia-smi returned no GPU memory")
        memory = int(values[0]) * 1024**2
        if memory <= 0:
            raise RuntimeError(f"nvidia-smi returned invalid GPU memory: {memory}")
        return memory

    @staticmethod
    def _probe_import(module_name: str) -> bool:
        """Check if a Python module is importable."""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# Stage-split topology planning.
#
# Extracted from bench_throughput.py::_plan_stage_split_topology so the
# benchmark CLI and the production orchestrator share one rule set.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageTopology:
    """Topology result from the planner.

    Layout: ``source → parse → encode``.

    In the Ray Data pipeline, ``encoders_per_gpu`` and ``gpu_fraction``
    are actively used for GPU planning.

    Attributes:
        encoders_per_gpu: EncodeUDF instances sharing each physical GPU.
            Always 1 for CPU backend; ≥1 for GPU backend.
        gpu_fraction: ``num_gpus=...`` fraction each EncodeUDF instance
            requests. ``1.0 / encoders_per_gpu`` for GPU; ``0.0`` for CPU.
        vram_per_actor: Soft VRAM budget per EncodeUDF in bytes.
        object_store_memory_bytes: Ray Plasma object-store budget in bytes.
    """

    encoders_per_gpu: int
    gpu_fraction: float
    vram_per_actor: int
    object_store_memory_bytes: int
    parse_actor_max_size: int
    encode_actor_max_size: int
    read_concurrency: int | None
    write_concurrency: int | None
    read_task_num_cpus: float
    map_actor_num_cpus: float
    write_task_num_cpus: float


class StageTopologyPlanner:
    """Computes a :class:`StageTopology` from cluster resources + backend."""

    _CPU_READ_WEIGHT = 2
    _CPU_PARSE_WEIGHT = 2
    _CPU_ENCODE_WEIGHT = 1
    _CPU_WRITE_WEIGHT = 2
    _CPU_HEADROOM_WEIGHT = 1
    DEFAULT_READ_TASK_NUM_CPUS = 0.5
    _MAP_ACTOR_NUM_CPUS = 1.0
    # Headroom factor: leave 15% of VRAM for driver/runtime/headroom.
    _VRAM_HEADROOM: float = 0.85
    # Default rows-per-file estimate used when sizing batch_size.
    _ROWS_PER_FILE_ESTIMATE: int = 40_000
    # Cap for auto-inferred encoders_per_gpu (conservative — plan task 5
    # notes 4 is the safe default for A800 80 GB).
    _ENCODERS_PER_GPU_AUTO_CAP: int = 4

    @classmethod
    def _cpu_budget_weight(cls) -> int:
        return (
            cls._CPU_READ_WEIGHT
            + cls._CPU_PARSE_WEIGHT
            + cls._CPU_ENCODE_WEIGHT
            + cls._CPU_WRITE_WEIGHT
            + cls._CPU_HEADROOM_WEIGHT
        )

    @classmethod
    def _concurrency_for_cpu_budget(
        cls,
        cpu_budget: float,
        task_num_cpus: float,
    ) -> int:
        concurrency = int(cpu_budget // task_num_cpus)
        if concurrency <= 0:
            raise ValueError(
                "stage CPU budget cannot schedule one task: "
                f"cpu_budget={cpu_budget}, task_num_cpus={task_num_cpus}"
            )
        return concurrency

    @classmethod
    def _plan_cpu_stage_defaults(
        cls,
        cluster_cpus: int,
        read_task_num_cpus: float,
        write_task_num_cpus: float,
    ) -> tuple[int, int, int, int]:
        budget_unit = cluster_cpus // cls._cpu_budget_weight()
        if budget_unit <= 0:
            raise ValueError(
                "automatic CPU topology requires at least "
                f"{cls._cpu_budget_weight()} CPUs, got {cluster_cpus}"
            )
        return (
            cls._concurrency_for_cpu_budget(
                budget_unit * cls._CPU_READ_WEIGHT,
                read_task_num_cpus,
            ),
            cls._concurrency_for_cpu_budget(
                budget_unit * cls._CPU_PARSE_WEIGHT,
                cls._MAP_ACTOR_NUM_CPUS,
            ),
            cls._concurrency_for_cpu_budget(
                budget_unit * cls._CPU_ENCODE_WEIGHT,
                cls._MAP_ACTOR_NUM_CPUS,
            ),
            cls._concurrency_for_cpu_budget(
                budget_unit * cls._CPU_WRITE_WEIGHT,
                write_task_num_cpus,
            ),
        )

    @classmethod
    def _plan_gpu_stage_defaults(
        cls,
        *,
        cluster_cpus: int,
        encode_actor_capacity: int,
        read_task_num_cpus: float,
        write_task_num_cpus: float,
    ) -> tuple[int, int, int]:
        headroom = max(1, cluster_cpus // cls._cpu_budget_weight())
        encode_cpu_budget = encode_actor_capacity * cls._MAP_ACTOR_NUM_CPUS
        shared_cpu_budget = cluster_cpus - headroom - encode_cpu_budget
        shared_budget_unit = shared_cpu_budget // 3
        if shared_budget_unit <= 0:
            raise ValueError(
                "GPU encode actors leave insufficient CPU capacity for read, parse, "
                f"and write stages: cluster_cpus={cluster_cpus}, "
                f"encode_actors={encode_actor_capacity}, headroom={headroom}"
            )
        return (
            cls._concurrency_for_cpu_budget(
                shared_budget_unit,
                read_task_num_cpus,
            ),
            cls._concurrency_for_cpu_budget(
                shared_budget_unit,
                cls._MAP_ACTOR_NUM_CPUS,
            ),
            cls._concurrency_for_cpu_budget(
                shared_budget_unit,
                write_task_num_cpus,
            ),
        )

    @classmethod
    def _validate_cpu_allocation(
        cls,
        *,
        cluster_cpus: int,
        read_concurrency: int,
        read_task_num_cpus: float,
        parse_actor_max_size: int,
        encode_actor_max_size: int,
        write_concurrency: int,
        write_task_num_cpus: float,
    ) -> None:
        allocated = (
            read_concurrency * read_task_num_cpus
            + parse_actor_max_size * cls._MAP_ACTOR_NUM_CPUS
            + encode_actor_max_size * cls._MAP_ACTOR_NUM_CPUS
            + write_concurrency * write_task_num_cpus
        )
        if allocated > cluster_cpus:
            raise ValueError(
                "stage concurrency exceeds cluster CPU capacity: "
                f"allocated={allocated}, cluster_cpus={cluster_cpus}"
            )

    def infer_encoders_per_gpu(self, gpu_memory_bytes: int) -> int:
        """Auto-infer ``encoders_per_gpu`` from per-GPU VRAM.

        Heuristic: ~6 GiB per EncodeUDF instance (covers ResourceRequirements
        4 GiB nominal + GPU mempool + driver headroom). Capped at
        :attr:`_ENCODERS_PER_GPU_AUTO_CAP` so multi-tenant clusters stay
        predictable; explicit CLI/env overrides bypass this cap.
        """
        if gpu_memory_bytes <= 0:
            raise ValueError(
                f"gpu_memory_bytes must be positive, got {gpu_memory_bytes}"
            )
        gb = gpu_memory_bytes / (1024**3)
        return max(1, min(int(gb // 6), self._ENCODERS_PER_GPU_AUTO_CAP))

    def plan(
        self,
        backend: str,
        cluster_cpus: int,
        n_gpus: int,
        object_store_capacity_bytes: int,
        encoders_per_gpu: int | None = None,
        gpu_memory_bytes: int = 0,
        concurrent_pipelines: int = 1,
        read_concurrency: int | None = None,
        read_task_num_cpus: float | None = None,
        write_concurrency: int | None = None,
        write_task_num_cpus: float | None = None,
        parse_actor_max_size: int | None = None,
        encode_actor_max_size: int | None = None,
        object_store_memory_fraction: float = 0.7,
    ) -> StageTopology:
        """Compute a :class:`StageTopology` for the requested backend.

        Args:
            backend: ``"cpu"`` or ``"gpu"``.
            cluster_cpus: Total CPU cores visible to Ray.
            n_gpus: Total GPUs visible to Ray (ignored for cpu backend).
            encoders_per_gpu: GPU backend only — actors per physical GPU.
                ``None`` (default) auto-infers from ``gpu_memory_bytes``.
            gpu_memory_bytes: Per-GPU VRAM (used for auto encoders_per_gpu
                + ``vram_per_actor`` budget). Ignored for cpu backend.
            object_store_capacity_bytes: Total Ray object-store capacity.

        Returns:
            Fully populated :class:`StageTopology`.
        """
        if cluster_cpus <= 0:
            raise ValueError(f"cluster_cpus must be positive, got {cluster_cpus}")
        if backend not in {"cpu", "gpu"}:
            raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")
        if n_gpus < 0:
            raise ValueError(f"n_gpus must be non-negative, got {n_gpus}")
        if object_store_capacity_bytes <= 0:
            raise ValueError(
                "object_store_capacity_bytes must be positive, "
                f"got {object_store_capacity_bytes}"
            )
        if encoders_per_gpu is not None and encoders_per_gpu <= 0:
            raise ValueError(
                f"encoders_per_gpu must be positive, got {encoders_per_gpu}"
            )
        if concurrent_pipelines <= 0:
            raise ValueError(
                f"concurrent_pipelines must be positive, got {concurrent_pipelines}"
            )
        if not 0.0 < object_store_memory_fraction <= 1.0:
            raise ValueError(
                "object_store_memory_fraction must be in (0, 1], "
                f"got {object_store_memory_fraction}"
            )
        for name, value in (
            ("read_concurrency", read_concurrency),
            ("read_task_num_cpus", read_task_num_cpus),
            ("write_concurrency", write_concurrency),
            ("write_task_num_cpus", write_task_num_cpus),
            ("parse_actor_max_size", parse_actor_max_size),
            ("encode_actor_max_size", encode_actor_max_size),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")

        planned_write_task_num_cpus = (
            DEFAULT_WRITE_TASK_NUM_CPUS
            if write_task_num_cpus is None
            else write_task_num_cpus
        )
        planned_read_task_num_cpus = (
            self.DEFAULT_READ_TASK_NUM_CPUS
            if read_task_num_cpus is None
            else read_task_num_cpus
        )

        object_store_memory_bytes = (
            int(object_store_capacity_bytes * object_store_memory_fraction)
            // concurrent_pipelines
        )
        if object_store_memory_bytes <= 0:
            raise ValueError(
                "object-store budget must be positive after pipeline allocation"
            )

        if backend == "gpu":
            if n_gpus <= 0:
                raise ValueError(
                    "StageTopologyPlanner.plan(backend='gpu') requires n_gpus > 0"
                )
            if gpu_memory_bytes <= 0:
                raise ValueError(
                    f"gpu_memory_bytes must be positive, got {gpu_memory_bytes}"
                )
            if encoders_per_gpu is None:
                encoders_per_gpu = self.infer_encoders_per_gpu(gpu_memory_bytes)
            gpu_encode_actor_capacity = n_gpus * encoders_per_gpu
            (
                default_read_concurrency,
                default_parse_actor_max_size,
                default_write_concurrency,
            ) = self._plan_gpu_stage_defaults(
                cluster_cpus=cluster_cpus,
                encode_actor_capacity=gpu_encode_actor_capacity,
                read_task_num_cpus=planned_read_task_num_cpus,
                write_task_num_cpus=planned_write_task_num_cpus,
            )
            planned_read_concurrency = read_concurrency or default_read_concurrency
            planned_parse_actor_max_size = (
                parse_actor_max_size or default_parse_actor_max_size
            )
            planned_encode_actor_max_size = (
                encode_actor_max_size or gpu_encode_actor_capacity
            )
            planned_write_concurrency = write_concurrency or default_write_concurrency
            if planned_encode_actor_max_size > gpu_encode_actor_capacity:
                raise ValueError(
                    "encode_actor_max_size exceeds fractional GPU capacity: "
                    f"{planned_encode_actor_max_size} > {gpu_encode_actor_capacity}"
                )
            self._validate_cpu_allocation(
                cluster_cpus=cluster_cpus,
                read_concurrency=planned_read_concurrency,
                read_task_num_cpus=planned_read_task_num_cpus,
                parse_actor_max_size=planned_parse_actor_max_size,
                encode_actor_max_size=planned_encode_actor_max_size,
                write_concurrency=planned_write_concurrency,
                write_task_num_cpus=planned_write_task_num_cpus,
            )

            gpu_fraction = 1.0 / encoders_per_gpu
            vram_per_actor = (
                int(gpu_memory_bytes * self._VRAM_HEADROOM / encoders_per_gpu)
                if gpu_memory_bytes > 0
                else 0
            )

            return StageTopology(
                encoders_per_gpu=encoders_per_gpu,
                gpu_fraction=gpu_fraction,
                vram_per_actor=vram_per_actor,
                object_store_memory_bytes=object_store_memory_bytes,
                parse_actor_max_size=planned_parse_actor_max_size,
                encode_actor_max_size=planned_encode_actor_max_size,
                read_concurrency=planned_read_concurrency,
                write_concurrency=planned_write_concurrency,
                read_task_num_cpus=planned_read_task_num_cpus,
                map_actor_num_cpus=self._MAP_ACTOR_NUM_CPUS,
                write_task_num_cpus=planned_write_task_num_cpus,
            )

        (
            default_read_concurrency,
            default_parse_actor_max_size,
            default_encode_actor_max_size,
            default_write_concurrency,
        ) = self._plan_cpu_stage_defaults(
            cluster_cpus,
            planned_read_task_num_cpus,
            planned_write_task_num_cpus,
        )
        planned_read_concurrency = read_concurrency or default_read_concurrency
        planned_parse_actor_max_size = (
            parse_actor_max_size or default_parse_actor_max_size
        )
        planned_encode_actor_max_size = (
            encode_actor_max_size or default_encode_actor_max_size
        )
        planned_write_concurrency = write_concurrency or default_write_concurrency
        self._validate_cpu_allocation(
            cluster_cpus=cluster_cpus,
            read_concurrency=planned_read_concurrency,
            read_task_num_cpus=planned_read_task_num_cpus,
            parse_actor_max_size=planned_parse_actor_max_size,
            encode_actor_max_size=planned_encode_actor_max_size,
            write_concurrency=planned_write_concurrency,
            write_task_num_cpus=planned_write_task_num_cpus,
        )
        return StageTopology(
            encoders_per_gpu=1,
            gpu_fraction=0.0,
            vram_per_actor=0,
            object_store_memory_bytes=object_store_memory_bytes,
            parse_actor_max_size=planned_parse_actor_max_size,
            encode_actor_max_size=planned_encode_actor_max_size,
            read_concurrency=planned_read_concurrency,
            write_concurrency=planned_write_concurrency,
            read_task_num_cpus=planned_read_task_num_cpus,
            map_actor_num_cpus=self._MAP_ACTOR_NUM_CPUS,
            write_task_num_cpus=planned_write_task_num_cpus,
        )


@dataclass(frozen=True, slots=True)
class WorkerMemoryPlan:
    """Heap reservations and concurrency limits for one streaming pipeline.

    Ray's object-store budget does not account for Python/C++ heap owned by
    datasource tasks and actor workers.  This plan reserves that heap as Ray
    resources and derives stage concurrency from the same finite budget.
    """

    heap_budget_bytes: int
    read_task_memory_bytes: int
    parse_actor_memory_bytes: int
    encode_actor_memory_bytes: int
    read_concurrency: int
    parse_actor_max_size: int
    encode_actor_max_size: int


class WorkerMemoryPlanner:
    """Bound worker heap and stage fan-out without CPU-count over-scaling."""

    _HEAP_BUDGET_FRACTION = 0.45
    _ACTOR_HEAP_OVERHEAD_BYTES = 512 * 1024 * 1024

    def plan(
        self,
        *,
        cluster_cpus: int,
        cluster_memory_bytes: int,
        read_concurrency: int,
        parse_actor_max_size: int,
        encode_actor_max_size: int,
        read_task_memory_bytes: int,
        source_block_bytes: int,
        target_block_bytes: int,
        read_concurrency_explicit: bool,
        parse_actor_max_explicit: bool,
        encode_actor_max_explicit: bool,
    ) -> WorkerMemoryPlan:
        values = {
            "cluster_cpus": cluster_cpus,
            "read_concurrency": read_concurrency,
            "parse_actor_max_size": parse_actor_max_size,
            "encode_actor_max_size": encode_actor_max_size,
            "read_task_memory_bytes": read_task_memory_bytes,
            "source_block_bytes": source_block_bytes,
            "target_block_bytes": target_block_bytes,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in (
            ("read_concurrency", read_concurrency),
            ("parse_actor_max_size", parse_actor_max_size),
            ("encode_actor_max_size", encode_actor_max_size),
        ):
            if value > cluster_cpus:
                raise ValueError(
                    f"{name} cannot exceed cluster_cpus: {value} > {cluster_cpus}"
                )
        if cluster_memory_bytes <= 0:
            raise ValueError(
                f"cluster_memory_bytes must be positive, got {cluster_memory_bytes}"
            )

        heap_budget = int(cluster_memory_bytes * self._HEAP_BUDGET_FRACTION)
        actor_memory = max(
            self._ACTOR_HEAP_OVERHEAD_BYTES,
            source_block_bytes + (2 * target_block_bytes),
        )
        # Keep one read task and one actor from making a valid cluster
        # configuration impossible.  The two actor stages share this budget.
        minimum_required = read_task_memory_bytes + (2 * actor_memory)
        if heap_budget < minimum_required:
            raise ValueError(
                "worker heap budget is too small for one streaming pipeline: "
                f"budget={heap_budget}, required={minimum_required}"
            )

        read_budget = heap_budget // 3
        read_cap = max(1, min(cluster_cpus, read_budget // read_task_memory_bytes))
        if read_concurrency_explicit and read_concurrency > read_cap:
            raise ValueError(
                "read_concurrency exceeds the worker heap budget: "
                f"requested={read_concurrency}, maximum={read_cap}"
            )
        bounded_reads = min(read_concurrency, read_cap)

        stage_budget = heap_budget - (bounded_reads * read_task_memory_bytes)
        stage_cap = max(1, min(cluster_cpus, stage_budget // (2 * actor_memory)))
        if parse_actor_max_explicit and parse_actor_max_size > stage_cap:
            raise ValueError(
                "parse_actor_max_size exceeds the worker heap budget: "
                f"requested={parse_actor_max_size}, maximum={stage_cap}"
            )
        if encode_actor_max_explicit and encode_actor_max_size > stage_cap:
            raise ValueError(
                "encode_actor_max_size exceeds the worker heap budget: "
                f"requested={encode_actor_max_size}, maximum={stage_cap}"
            )

        return WorkerMemoryPlan(
            heap_budget_bytes=heap_budget,
            read_task_memory_bytes=read_task_memory_bytes,
            parse_actor_memory_bytes=actor_memory,
            encode_actor_memory_bytes=actor_memory,
            read_concurrency=min(read_concurrency, read_cap),
            parse_actor_max_size=min(parse_actor_max_size, stage_cap),
            encode_actor_max_size=min(encode_actor_max_size, stage_cap),
        )


@dataclass(frozen=True, slots=True)
class StreamingMemoryPlan:
    """Object-store sizing for one Ray Data streaming pipeline.

    The plan accounts for every source task's current block plus its bounded
    prefetch buffers and reserves one downstream block.  Values explicitly
    supplied by a caller are retained; only ``None`` fields are inferred.
    """

    source_csv_block_size_bytes: int
    target_min_block_size_bytes: int
    target_max_block_size_bytes: int
    datasource_prefetch_buffer_size_bytes: int


class StreamingMemoryPlanner:
    """Derive safe Ray Data block sizes from the available object store."""

    _MIN_AUTO_BLOCK_BYTES = 8 * 1024 * 1024
    _MAX_AUTO_BLOCK_BYTES = 128 * 1024 * 1024

    def plan(
        self,
        *,
        object_store_memory_bytes: int,
        read_concurrency: int,
        datasource_prefetch_max_buffers: int,
        source_csv_block_size_bytes: int | None,
        target_min_block_size_bytes: int | None,
        target_max_block_size_bytes: int | None,
        datasource_prefetch_buffer_size_bytes: int | None,
    ) -> StreamingMemoryPlan:
        if object_store_memory_bytes <= 0:
            raise ValueError(
                "automatic streaming block sizing requires a positive object-store "
                f"budget, got {object_store_memory_bytes}"
            )
        if read_concurrency <= 0:
            raise ValueError(
                f"read_concurrency must be positive, got {read_concurrency}"
            )
        if datasource_prefetch_max_buffers <= 0:
            raise ValueError(
                "datasource_prefetch_max_buffers must be positive, "
                f"got {datasource_prefetch_max_buffers}"
            )

        auto_fields = sum(
            value is None
            for value in (
                source_csv_block_size_bytes,
                target_min_block_size_bytes,
                target_max_block_size_bytes,
                datasource_prefetch_buffer_size_bytes,
            )
        )
        if auto_fields:
            slots = read_concurrency * (datasource_prefetch_max_buffers + 1) + 2
            per_slot_budget = object_store_memory_bytes // slots
            minimum_resident = (
                read_concurrency
                * self._MIN_AUTO_BLOCK_BYTES
                * (datasource_prefetch_max_buffers + 1)
                + self._MIN_AUTO_BLOCK_BYTES
            )
            if object_store_memory_bytes < minimum_resident:
                raise ValueError(
                    "object-store budget is too small for bounded streaming blocks: "
                    f"budget={object_store_memory_bytes}, read_concurrency={read_concurrency}, "
                    f"prefetch_buffers={datasource_prefetch_max_buffers}"
                )
            auto_source = min(
                self._MAX_AUTO_BLOCK_BYTES,
                max(self._MIN_AUTO_BLOCK_BYTES, per_slot_budget // 3),
            )
            auto_prefetch = min(
                auto_source,
                max(self._MIN_AUTO_BLOCK_BYTES, per_slot_budget // 3),
            )
            auto_target_max = min(
                self._MAX_AUTO_BLOCK_BYTES,
                max(self._MIN_AUTO_BLOCK_BYTES, per_slot_budget),
            )
            auto_target_min = max(
                self._MIN_AUTO_BLOCK_BYTES,
                auto_target_max // 2,
            )
        else:
            auto_source = auto_prefetch = auto_target_max = auto_target_min = 0

        source = (
            auto_source
            if source_csv_block_size_bytes is None
            else source_csv_block_size_bytes
        )
        target_max = (
            auto_target_max
            if target_max_block_size_bytes is None
            else target_max_block_size_bytes
        )
        target_min = (
            auto_target_min
            if target_min_block_size_bytes is None
            else target_min_block_size_bytes
        )
        prefetch = (
            auto_prefetch
            if datasource_prefetch_buffer_size_bytes is None
            else datasource_prefetch_buffer_size_bytes
        )
        for name, value in (
            ("source_csv_block_size_bytes", source),
            ("target_min_block_size_bytes", target_min),
            ("target_max_block_size_bytes", target_max),
            ("datasource_prefetch_buffer_size_bytes", prefetch),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if target_min > target_max:
            raise ValueError(
                "target_min_block_size_bytes must be <= target_max_block_size_bytes, "
                f"got {target_min} > {target_max}"
            )

        if auto_fields:
            estimated_resident = (
                read_concurrency * (source + prefetch * datasource_prefetch_max_buffers)
                + target_max
            )
            if estimated_resident > object_store_memory_bytes:
                raise ValueError(
                    "automatically sized streaming blocks exceed the object-store "
                    f"budget: resident={estimated_resident}, budget={object_store_memory_bytes}"
                )

        return StreamingMemoryPlan(
            source_csv_block_size_bytes=source,
            target_min_block_size_bytes=target_min,
            target_max_block_size_bytes=target_max,
            datasource_prefetch_buffer_size_bytes=prefetch,
        )
