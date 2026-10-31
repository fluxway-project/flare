"""Finite heap-stream execution for recipe-managed production jobs.

Parsing, encoding, Parquet writing and transactional publication remain owned by
existing Flare components. Only finite task orchestration changes here.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import resource
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import ray

from flare.data.pipeline.production.publication.manifest import PublicationManifest
from flare.data.pipeline.production.publish.datasource import (
    _read_decompress_file,
    default_prefetch_reader_factory,
)
from flare.data.pipeline.production.publish.orchestrator import (
    BatchResult,
    RayDataPipeline,
)
from flare.data.pipeline.production.publish.stages.fused_parse_encode import (
    FusedParseEncodeUDF,
)
from flare.io.artifacts import write_json

ENGINE_NAME = "heap_stream_v3"


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def plan_heap_workers(
    *,
    cluster_cpus: int,
    cluster_memory_bytes: int,
    cpus_per_worker: int,
    memory_per_worker_bytes: int,
    requested: int | None,
) -> int:
    """Leave driver headroom and reserve no more than 90% of worker memory."""
    if cpus_per_worker <= 0 or memory_per_worker_bytes <= 0:
        raise ValueError("Heap worker CPU and memory reservations must be positive")
    reserved_cpus = max(1, math.ceil(cluster_cpus * 0.035))
    cpu_slots = (cluster_cpus - reserved_cpus) // cpus_per_worker
    memory_slots = int(cluster_memory_bytes * 0.9) // memory_per_worker_bytes
    available = min(cpu_slots, memory_slots)
    if available < 1:
        raise ValueError("No heap-stream worker fits the available Ray resources")
    if requested is not None and requested > available:
        raise ValueError(
            f"heap-workers exceeds available capacity: requested={requested}, maximum={available}"
        )
    return requested or available


def stream_group(
    paths, constructor, config, source_provider, datasink, max_rows_per_batch
):
    started = time.perf_counter()
    cpu_started = time.process_time()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(2)
    transform = FusedParseEncodeUDF(**constructor)
    source_rows = {}

    def encoded_blocks():
        context = (
            source_provider.open() if source_provider is not None else nullcontext(None)
        )
        with context as source:
            factory = partial(
                default_prefetch_reader_factory,
                buffer_size=config.datasource_prefetch_buffer_size_bytes,
                max_buffers=config.datasource_prefetch_max_buffers,
                **({"source": source} if source is not None else {}),
            )
            for path in paths:
                rows = 0
                for raw in _read_decompress_file(
                    path,
                    config.source_csv_block_size_bytes,
                    reader_factory=factory,
                    source_partition_defaults=config.source_partition_defaults,
                ):
                    for offset in range(0, raw.num_rows, max_rows_per_batch):
                        batch = raw.slice(offset, max_rows_per_batch)
                        encoded = transform(batch)
                        if encoded.num_rows != batch.num_rows:
                            raise RuntimeError("Transform changed source row count")
                        rows += batch.num_rows
                        yield encoded
                source_rows[path] = rows

    written = datasink.write(encoded_blocks(), None)
    if sum(f.record_count for f in written.data_files) != sum(source_rows.values()):
        raise RuntimeError("Parquet record counts differ from decoded source rows")
    return (
        written,
        source_rows,
        {
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "code_sha256": code_sha256(),
            "node_id": ray.get_runtime_context().get_node_id(),
        },
    )


_remote_stream_group = ray.remote(stream_group)


class HeapStreamDataset:
    def __init__(
        self,
        paths,
        constructor_ref,
        config,
        source_provider,
        *,
        workers,
        cpus_per_worker,
        memory_per_worker,
        audit_path,
        source_sizes,
        max_rows_per_batch,
    ):
        self.paths = paths
        self.constructor_ref = constructor_ref
        self.config = config
        self.source_provider = source_provider
        self.workers = workers
        self.cpus_per_worker = cpus_per_worker
        self.memory_per_worker = memory_per_worker
        self.audit_path = audit_path
        self.source_sizes = source_sizes
        self.max_rows_per_batch = max_rows_per_batch
        self.total_rows = 0

    def write_datasink(self, datasink, *, ray_remote_args=None, concurrency=None):
        # Source and sink working sets are bounded per task; only metadata is
        # returned to the driver. The original sink still owns the one commit.
        count = min(len(self.paths), self.workers)
        groups = [[] for _ in range(count)]
        loads = [(0, index) for index in range(count)]
        sizes = {
            path: max(1, self.source_sizes[path])
            if self.source_sizes is not None
            else 1
            for path in self.paths
        }
        for path in sorted(self.paths, key=lambda path: (-sizes[path], path)):
            size, index = heapq.heappop(loads)
            groups[index].append(path)
            heapq.heappush(loads, (size + sizes[path], index))
        groups = [tuple(sorted(group)) for group in groups]
        datasink.on_write_start()
        started = time.perf_counter()
        pending = [
            _remote_stream_group.options(
                num_cpus=self.cpus_per_worker,
                memory=self.memory_per_worker,
                max_retries=0,
                scheduling_strategy="SPREAD",
            ).remote(
                group,
                self.constructor_ref,
                self.config,
                self.source_provider,
                datasink,
                self.max_rows_per_batch,
            )
            for group in groups
        ]
        all_refs = list(pending)
        written = []
        source_rows = {}
        durations = []
        expected_code = code_sha256()
        try:
            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                result, counts, metrics = ray.get(ready[0])
                if source_rows.keys() & counts.keys():
                    raise RuntimeError("Source path processed by multiple groups")
                if metrics["code_sha256"] != expected_code:
                    raise RuntimeError("Execution code differs across workers")
                source_rows.update(counts)
                written.append(result)
                durations.append(metrics)
                if len(written) % 16 == 0 or not pending:
                    print(
                        f"Heap stream groups {len(written)}/{count}; decoded rows {sum(source_rows.values())}",
                        flush=True,
                    )
        except BaseException:
            for ref in all_refs:
                ray.cancel(ref, force=True)
            raise
        if set(source_rows) != set(self.paths):
            raise RuntimeError("Source group coverage is incomplete")
        self.total_rows = sum(source_rows.values())
        if self.audit_path is not None:
            report = {
                "source_rows": source_rows,
                "execution_engine": ENGINE_NAME,
                "code_sha256": expected_code,
                "worker_nodes": sorted({item["node_id"] for item in durations}),
                "total_rows": self.total_rows,
                "groups": count,
                "worker_seconds": sum(item["wall_seconds"] for item in durations),
                "worker_cpu_seconds": sum(item["cpu_seconds"] for item in durations),
                "max_worker_rss_bytes": max(
                    item["max_rss_bytes"] for item in durations
                ),
                "elapsed_before_commit": time.perf_counter() - started,
                "data_files": sum(len(x.data_files) for x in written),
            }
            write_json(self.audit_path, report)
        datasink.on_write_complete(SimpleNamespace(write_returns=written))


class HeapStreamPipeline(RayDataPipeline):
    def __init__(
        self,
        *args,
        heap_workers: int,
        heap_cpus_per_worker: int,
        heap_memory_per_worker: int,
        heap_audit_dir: Path,
        **kwargs,
    ):
        artifact = kwargs.get("artifact")
        if artifact is None:
            raise ValueError("Heap streaming requires a compiled artifact")
        super().__init__(*args, **kwargs)
        self._heap_source = kwargs.get("source_provider")
        self._heap_workers = heap_workers
        self._heap_cpus = heap_cpus_per_worker
        self._heap_memory = heap_memory_per_worker
        self._heap_audit_dir = Path(heap_audit_dir)
        self._heap_constructor_ref = ray.put(
            {
                "parser_spec": artifact.parser_spec,
                "metadata_maps": {
                    name: dict(values)
                    for name, values in artifact.metadata_maps.items()
                },
                "encoded_plan": artifact.encoded_plan,
                "output_schema": artifact.compiled_pipeline.output_schema,
            }
        )

    def execute(self, file_paths, terminal):
        started = time.perf_counter()
        sizes_for = getattr(self._heap_source, "file_sizes", None)
        source_sizes = sizes_for(file_paths) if callable(sizes_for) else None
        audit_path = self._heap_audit_dir / (
            PublicationManifest.from_files(file_paths).identity.value + ".json"
        )
        dataset = HeapStreamDataset(
            file_paths,
            self._heap_constructor_ref,
            self.config,
            self._heap_source,
            workers=self._heap_workers,
            cpus_per_worker=self._heap_cpus,
            memory_per_worker=self._heap_memory,
            audit_path=audit_path,
            source_sizes=source_sizes,
            max_rows_per_batch=self._max_rows_per_batch,
        )
        result = terminal.finalize(dataset, schema=self.output_schema)
        if result.total_rows != dataset.total_rows:
            raise RuntimeError("Committed records differ from source audit")
        return BatchResult(
            total_rows=result.total_rows,
            elapsed_seconds=time.perf_counter() - started,
            outputs=result.outputs,
        )
