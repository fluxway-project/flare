from __future__ import annotations

import time
from dataclasses import make_dataclass
from typing import Any

import jax.tree_util as jtu
import numpy as np
from fluxway.common.decorators import data_type, requires_worker_process
from fluxway.common.operator_policy import AsyncPolicy

from flare.data.pipeline.consumption.source.batching import BatchValidity
from flare.runtime.fluxway.resources import normalize_resource_vector
from flare.runtime.fluxway.transport import transport_payload
from flare.training.components.metrics.aggregation import (
    MetricWindow,
)
from flare.training.components.metrics.contracts import MetricResult, MetricSink
from recipes.recsys.ranking.metrics.ray_data import RayDataMetricsComputer
from recipes.recsys.ranking.metrics.sink import (
    RankingTensorBoardMetricSink,
)

_train_metrics_out_fields: list[tuple[str, type]] = []
TrainMetricsOutput = make_dataclass("TrainMetricsOutput", _train_metrics_out_fields)
TrainMetricsOutput = data_type()(TrainMetricsOutput)
jtu.register_dataclass(TrainMetricsOutput, data_fields=[], meta_fields=[])


class TrainMetricsComputer:
    def __init__(
        self,
        *,
        sink: MetricSink,
        metrics: RayDataMetricsComputer,
    ) -> None:
        self._sink = sink
        self._metrics = metrics

    def put_batch(
        self,
        *,
        scores: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> Any:
        return self._metrics.put_batch(
            scores=scores,
            labels=labels,
            groups=groups,
        )

    def compute_and_write(self, window: MetricWindow) -> dict[str, Any]:
        per_target_metrics = (
            self._metrics.compute(window.prediction_refs)
            if window.prediction_refs
            else {}
        )
        return self.write_metrics(window, per_target_metrics)

    def submit_metrics(self, refs: tuple[Any, ...]) -> Any:
        return self._metrics.submit_compute(refs)

    def close(self) -> None:
        self._sink.close()

    def write_metrics(
        self,
        window: MetricWindow,
        per_target_metrics: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if window.prediction_refs and not per_target_metrics:
            raise RuntimeError("metric computation returned no target metrics")
        if window.prediction_refs:
            sample_counts = {
                int(metrics["sample_count"]) for metrics in per_target_metrics.values()
            }
            if len(sample_counts) != 1:
                raise RuntimeError(
                    f"training targets disagree on sample count: {sample_counts}"
                )
            metric_sample_count = sample_counts.pop()
            loss = window.loss_sum / window.sampled_step_count
        else:
            metric_sample_count = 0
            loss = None
        throughput_steps = window.throughput_steps
        throughput_samples = window.throughput_samples

        values: dict[str, float] = {
            "metric_sample_count": float(metric_sample_count),
            "throughput_steps_per_sec": float(throughput_steps),
            "throughput_samples_per_sec": float(throughput_samples),
        }
        if loss is not None:
            values["loss"] = float(loss)
        for target_name, metrics in per_target_metrics.items():
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    values[f"{target_name}/{name}"] = float(value)
        self._sink.write(
            MetricResult(
                values=values,
                metadata={
                    "global_step": window.global_step,
                    "trained_sample_count": window.trained_sample_count,
                    "sampled_step_count": window.sampled_step_count,
                },
            )
        )

        loss_text = "none" if loss is None else f"{loss:.6f}"
        print(
            "[TrainMetrics] "
            f"step={window.global_step} "
            f"loss={loss_text} "
            f"trained_samples={window.trained_sample_count} "
            f"metric_samples={metric_sample_count} "
            f"sampled_steps={window.sampled_step_count} "
            f"throughput={throughput_samples:.2f} samples/s",
            flush=True,
        )
        return {
            "global_step": window.global_step,
            "loss": loss,
            "metric_sample_count": metric_sample_count,
            "throughput_samples_per_sec": throughput_samples,
            "per_target_metrics": per_target_metrics,
        }


@requires_worker_process
class TrainMetricsAggregatorSPMD:
    rank: int
    world_size: int
    ctx: Any
    rank0_address: str
    ray_component_limits: Any

    def __init__(
        self,
        log_dir: str,
        target_labels: np.ndarray,
        target_names: tuple[str, ...],
        window_steps: int = 100,
        negative_factor: float | None = None,
        target_step: int | None = None,
    ) -> None:
        if window_steps <= 0:
            raise ValueError(f"window_steps must be positive, got {window_steps}")
        if target_step is not None and target_step <= 0:
            raise ValueError(f"target_step must be positive, got {target_step}")
        self.window_steps = int(window_steps)
        self.target_step = None if target_step is None else int(target_step)
        self._target_labels = np.asarray(target_labels, dtype=np.float32)
        self._computer = TrainMetricsComputer(
            sink=RankingTensorBoardMetricSink(log_dir),
            metrics=RayDataMetricsComputer(
                target_names=tuple(target_names),
                negative_factor=negative_factor,
                ray_component_limits=self.ray_component_limits,
                resource_normalizer=normalize_resource_vector,
            ),
        )
        self._prediction_refs: list[Any] = []
        self._loss_sum = 0.0
        self._sampled_step_count = 0
        self._step_count = 0
        self._trained_sample_count = 0
        self._global_step = 0
        self._window_start_time: float | None = None
        self.TrainMetricsOutput = TrainMetricsOutput

    async def on_shutdown(self) -> None:
        await self.wait_for_metrics()
        self._computer.close()

    async def step(self, **kwargs: Any) -> Any:
        scores = np.asarray(transport_payload(kwargs["scores"]), dtype=np.float32)
        labels = np.asarray(transport_payload(kwargs["labels"]), dtype=np.int32)
        groups = np.asarray(transport_payload(kwargs["group_ids"]), dtype=np.int64)
        loss = np.asarray(transport_payload(kwargs["loss"]), dtype=np.float32).reshape(
            -1
        )
        sampled_flags = np.asarray(
            transport_payload(kwargs["metric_sampled"]), dtype=np.int32
        ).reshape(-1)
        count_values = np.asarray(
            transport_payload(kwargs["samples_valid_count"])
        ).reshape(-1)
        if count_values.dtype != np.int32 or count_values.size != 1:
            raise ValueError("samples_valid_count must be one int32 scalar")
        validity = BatchValidity(labels.shape[0], int(count_values[0]))
        if groups.shape[0] != validity.batch_size:
            raise ValueError("group_ids must be row-aligned with labels")
        if sampled_flags.size == 0:
            raise ValueError("metric_sampled must be a non-empty static tensor")
        if not np.all(sampled_flags == sampled_flags[0]):
            raise ValueError(
                "metric_sampled must agree across RecsysBackboneTrain ranks, got "
                f"{sampled_flags.tolist()}"
            )
        if sampled_flags[0] not in (0, 1):
            raise ValueError(
                f"metric_sampled must be 0 or 1, got {int(sampled_flags[0])}"
            )
        sampled = bool(sampled_flags[0])
        if sampled:
            if loss.size == 0:
                raise ValueError("sampled TrainMetrics step requires a loss value")
            if scores.shape[0] != validity.batch_size:
                raise ValueError("sampled scores must be row-aligned with labels")
            labels = self._target_labels[:, labels].T.astype(np.int8, copy=False)
            self._prediction_refs.append(
                self._computer.put_batch(
                    scores=scores[: validity.samples_valid_count],
                    labels=labels[: validity.samples_valid_count],
                    groups=groups[: validity.samples_valid_count],
                )
            )
            self._loss_sum += float(loss.mean())
            self._sampled_step_count += 1
        elif scores.shape[0] == 0 or loss.size == 0:
            raise ValueError(
                "unsampled TrainMetrics step must carry non-empty static placeholders"
            )
        self._step_count += 1
        self._trained_sample_count += validity.samples_valid_count
        if self._window_start_time is None:
            self._window_start_time = time.time()
        ctx_step = getattr(getattr(self, "ctx", None), "step", None)
        if ctx_step is None:
            self._global_step += 1
        else:
            self._global_step = int(ctx_step) + 1
        should_submit = self._step_count >= self.window_steps
        if self.target_step is not None:
            should_submit = should_submit or self._global_step >= self.target_step
        if should_submit:
            await self._submit_window()
        return self.TrainMetricsOutput()

    async def wait_for_metrics(self) -> None:
        if self._step_count > 0:
            await self._submit_window()

    async def _submit_window(self) -> None:
        if self._step_count <= 0:
            raise RuntimeError(
                "TrainMetricsAggregatorSPMD cannot submit an empty window"
            )
        if self._window_start_time is None:
            raise RuntimeError(
                "TrainMetricsAggregatorSPMD window start time was not initialized"
            )
        window = MetricWindow(
            prediction_refs=tuple(self._prediction_refs),
            loss_sum=float(self._loss_sum),
            sampled_step_count=int(self._sampled_step_count),
            step_count=int(self._step_count),
            trained_sample_count=int(self._trained_sample_count),
            global_step=int(self._global_step),
            elapsed_sec=time.time() - self._window_start_time,
        )
        self._prediction_refs = []
        self._loss_sum = 0.0
        self._sampled_step_count = 0
        self._step_count = 0
        self._trained_sample_count = 0
        self._window_start_time = None

        per_target_metrics = (
            await self._computer.submit_metrics(window.prediction_refs)
            if window.prediction_refs
            else {}
        )
        self._computer.write_metrics(window, per_target_metrics)


def get_train_metrics_aggregator_spmd_class(
    *,
    input_sharding: dict[str, Any],
    runtime_env: dict[str, Any] | None,
    resource_profile: Any,
    max_inflight_steps: int,
) -> type:
    if max_inflight_steps <= 0:
        raise ValueError(
            f"max_inflight_steps must be positive, got {max_inflight_steps}"
        )
    from fluxway.common.bind import bind_spmd

    return bind_spmd(
        TrainMetricsAggregatorSPMD,
        resource_profile=resource_profile,
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(1,),
        runtime_env=runtime_env,
        step={
            "async_policy": AsyncPolicy.pipelined(depth=max_inflight_steps + 1),
            "input_sharding": input_sharding,
            "output_sharding": {},
        },
    )


__all__ = [
    "TrainMetricsAggregatorSPMD",
    "TrainMetricsComputer",
    "TrainMetricsOutput",
    "get_train_metrics_aggregator_spmd_class",
]
