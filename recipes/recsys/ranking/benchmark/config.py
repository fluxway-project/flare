from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DEFAULT_SOURCE_READ_CONCURRENCY = 8


@dataclass
class RankingThroughputBenchmarkConfigBase:
    samples_per_input_step: int = field(kw_only=True)
    steps: int = field(kw_only=True)
    batch_size: int = field(init=False)
    dp: int = 16
    reader_workers: int = 2
    jax_platform: str = "cpu"
    sparse_max_staleness_steps: int = 3
    sparse_workers: str = ""
    sparse_ckpt_backend: str | None = None
    worker_count: int = 0
    worker_buffer_size: int = 0
    source_read_concurrency: int = DEFAULT_SOURCE_READ_CONCURRENCY
    reader_repeat_first_batch: bool = False
    reader_execution_audit_path: str | None = None
    xla_cache_dir: str = "./xla"
    dataset_path: str | None = None
    completion_policy: str = "drop"
    storage_format: str = "parquet"
    catalog_name: str | None = None
    catalog_type: str | None = None
    catalog_uri: str | None = None
    warehouse: str | None = None
    iceberg_config: str | None = None
    iceberg_snapshot_id: int | None = None
    table_identifier: str | None = None
    feature_manifest_dir: str | None = None
    iceberg_max_open_files: int | None = None
    iceberg_poll_interval_s: float = 1.0
    ray_address: str | None = None
    ray_namespace: str | None = None
    ray_node_ip_address: str | None = None
    source_assignment_solve_timeout_s: float = 30.0
    source_assignment_policy: Literal[
        "none", "contiguous_only", "same_file_only", "both"
    ] = "none"
    s3_endpoint_url: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    file_storage_options: dict[str, Any] = field(default_factory=dict)
    profiler_trace_dir: str | None = None
    profiler_run_id: str | None = None
    profiler_buffer_mib: int = 64
    profiler_clock_sync_rounds: int = 8
    profiler_max_clock_uncertainty_us: int = 1000
    profiler_output_file: str | None = None
    profiler_diagnostics: list[str] = field(default_factory=list)
    profiler_skip_initial_steps: int = 4
    profiler_steady_state_only: bool = False
    null_index_sentinel: int = -1

    @property
    def global_batch_size(self) -> int:
        return self.batch_size

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps < 2:
            raise ValueError("throughput steps must be an integer of at least two")
        if self.jax_platform != "cpu":
            raise ValueError("CPU throughput benchmark requires jax_platform=cpu")
        if (
            not isinstance(self.samples_per_input_step, int)
            or isinstance(self.samples_per_input_step, bool)
            or self.samples_per_input_step <= 0
        ):
            raise ValueError("samples_per_input_step must be a positive integer")
        if not isinstance(self.dp, int) or isinstance(self.dp, bool) or self.dp <= 0:
            raise ValueError("dp must be a positive integer")
        if (
            not isinstance(self.reader_workers, int)
            or isinstance(self.reader_workers, bool)
            or self.reader_workers <= 0
        ):
            raise ValueError("reader_workers must be a positive integer")
        if self.reader_workers % self.dp:
            raise ValueError(
                "reader_workers must divide evenly by dp for whole-unit throughput"
            )
        self.batch_size = self.samples_per_input_step * self.dp

        if self.source_assignment_solve_timeout_s <= 0:
            raise ValueError("source_assignment_solve_timeout_s must be positive")
        if self.source_assignment_policy not in {
            "none",
            "contiguous_only",
            "same_file_only",
            "both",
        }:
            raise ValueError(
                "source_assignment_policy must be none, contiguous_only, same_file_only, or both"
            )
        if self.iceberg_max_open_files is not None and (
            not isinstance(self.iceberg_max_open_files, int)
            or isinstance(self.iceberg_max_open_files, bool)
            or self.iceberg_max_open_files <= 0
        ):
            raise ValueError("iceberg_max_open_files must be a positive integer")
        if self.iceberg_config is not None and any(
            value is not None
            for value in (
                self.catalog_name,
                self.catalog_type,
                self.catalog_uri,
                self.warehouse,
            )
        ):
            raise ValueError(
                "iceberg_config cannot be combined with individual catalog arguments"
            )
        if self.storage_format in {"parquet", "vortex"} and not self.dataset_path:
            raise ValueError(
                f"{self.storage_format.capitalize()} Reader requires --dataset_path"
            )
        if (
            self.storage_format in {"parquet", "vortex"}
            and self.dataset_path
            and self.dataset_path.startswith("s3://")
        ):
            self.file_storage_options = {
                "client_kwargs": {
                    "endpoint_url": self.s3_endpoint_url,
                    "aws_access_key_id": self.aws_access_key_id,
                    "aws_secret_access_key": self.aws_secret_access_key,
                }
            }


__all__ = [
    "DEFAULT_SOURCE_READ_CONCURRENCY",
    "RankingThroughputBenchmarkConfigBase",
]
