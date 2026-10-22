from __future__ import annotations

from dataclasses import make_dataclass
from typing import Any

import jax.tree_util as jtu
import numpy as np
from fluxway.common.decorators import data_type, requires_worker_process

from flare.data.pipeline.consumption.source.batching import BatchValidity
from flare.runtime.fluxway.resources import normalize_resource_vector
from flare.runtime.fluxway.transport import transport_payload
from recipes.recsys.ranking.metrics.ray_data import RayDataMetricsComputer

_metrics_out_fields: list[tuple[str, type]] = []
MetricsOutput = make_dataclass("MetricsOutput", _metrics_out_fields)
MetricsOutput = data_type()(MetricsOutput)
jtu.register_dataclass(MetricsOutput, data_fields=[], meta_fields=[])


@requires_worker_process
class MetricsAggregatorSPMD:
    rank: int
    world_size: int
    ctx: Any
    rank0_address: str
    ray_component_limits: Any

    def __init__(
        self,
        max_steps: int | None,
        target_labels: np.ndarray,
        target_names: tuple[str, ...],
        negative_factor: float | None = None,
        window_steps: int | None = None,
    ) -> None:
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self.max_steps = None if max_steps is None else int(max_steps)
        self.window_steps = (
            self.max_steps if window_steps is None else int(window_steps)
        )
        if self.window_steps is None or self.window_steps <= 0:
            raise ValueError("window_steps must be positive")
        self.target_names = tuple(target_names)
        self._target_labels = np.asarray(target_labels, dtype=np.float32)
        self._metrics = RayDataMetricsComputer(
            target_names=self.target_names,
            negative_factor=negative_factor,
            ray_component_limits=self.ray_component_limits,
            resource_normalizer=normalize_resource_vector,
        )
        self._prediction_refs: list[Any] = []
        self._current_step = 0
        self._window_start_step = 0

    async def on_shutdown(self) -> None:
        if self._prediction_refs:
            await self._submit_window()

    async def step(self, **kwargs: Any) -> Any:
        scores = np.asarray(transport_payload(kwargs["scores"]), dtype=np.float32)
        labels = np.asarray(transport_payload(kwargs["labels"]), dtype=np.int32)
        groups = np.asarray(transport_payload(kwargs["group_ids"]), dtype=np.int64)
        count_values = np.asarray(
            transport_payload(kwargs["samples_valid_count"])
        ).reshape(-1)
        if count_values.dtype != np.int32 or count_values.size != 1:
            raise ValueError("samples_valid_count must be one int32 scalar")
        validity = BatchValidity(scores.shape[0], int(count_values[0]))
        scores = scores[: validity.samples_valid_count]
        labels = labels[: validity.samples_valid_count]
        groups = groups[: validity.samples_valid_count]
        labels = self._target_labels[:, labels].T.astype(np.int8, copy=False)
        self._prediction_refs.append(
            self._metrics.put_batch(
                scores=scores,
                labels=labels,
                groups=groups,
            )
        )
        self._current_step += 1
        if self._current_step - self._window_start_step >= self.window_steps or (
            self.max_steps is not None and self._current_step >= self.max_steps
        ):
            await self._submit_window()
        return MetricsOutput()

    async def _submit_window(self) -> None:
        if not self._prediction_refs:
            raise RuntimeError("evaluation produced no metric batches")
        start, end = self._window_start_step + 1, self._current_step
        refs = tuple(self._prediction_refs)
        self._prediction_refs = []
        self._window_start_step = self._current_step
        results = await self._metrics.submit_compute(refs)
        self._report_results(results, start=start, end=end)

    def _report_results(
        self, results: dict[str, dict[str, Any]], *, start: int, end: int
    ) -> None:
        sample_counts = {
            int(results[target_name]["sample_count"])
            for target_name in self.target_names
        }
        if len(sample_counts) != 1:
            raise RuntimeError(
                f"evaluation targets disagree on sample count: {sample_counts}"
            )
        sample_count = sample_counts.pop()
        values = []
        for target_name in self.target_names:
            metrics = results[target_name]
            auc = f"{metrics['auc']:.4f}" if metrics["auc_defined"] else "undefined"
            gauc = (
                f"{metrics['grouped_auc']:.4f}"
                if metrics["grouped_auc_defined"]
                else "undefined"
            )
            values.append(f"AUC({target_name})={auc}")
            values.append(f"GAUC({target_name})={gauc}")
        print(
            "[MetricsAggregator] "
            f"scope=window window_steps={start}-{end} samples={sample_count}, "
            f"{', '.join(values)}",
            flush=True,
        )


def get_metrics_aggregator_spmd_class(
    input_sharding: dict[str, Any],
    runtime_env: dict[str, Any] | None,
    resource_profile: Any,
) -> type:
    from fluxway.common.bind import bind_spmd

    return bind_spmd(
        MetricsAggregatorSPMD,
        resource_profile=resource_profile,
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(1,),
        runtime_env=runtime_env,
        checkpoint_components=(),
        step={
            "input_sharding": input_sharding,
            "output_sharding": {},
        },
    )


__all__ = [
    "MetricsAggregatorSPMD",
    "MetricsOutput",
    "get_metrics_aggregator_spmd_class",
]
