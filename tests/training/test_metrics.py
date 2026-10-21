from __future__ import annotations

import asyncio
import importlib
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from fluxway import ResourceProfile

from flare.training.components.metrics.aggregation import MetricWindow
from flare.training.components.metrics.contracts import (
    MetricResult,
    MetricSink,
)
from flare.training.components.metrics.contracts import (
    MetricResult as TrainingMetricResult,
)
from flare.training.components.metrics.sinks.tensorboard import ConsoleMetricSink
from recipes.recsys.ranking.metrics.eval_metrics import (
    MetricsAggregatorSPMD,
    get_metrics_aggregator_spmd_class,
)
from recipes.recsys.ranking.metrics.ray_data import RayDataMetricsComputer
from recipes.recsys.ranking.metrics.train_metrics import (
    TrainMetricsAggregatorSPMD,
    get_train_metrics_aggregator_spmd_class,
)


def test_training_metrics_reuse_the_runtime_result_contract(capsys) -> None:
    assert TrainingMetricResult is MetricResult
    sink = ConsoleMetricSink()
    assert isinstance(sink, MetricSink)
    sink.write(MetricResult(values={"auc": 0.75}, metadata={"global_step": 7}))
    assert "[Step 7] Metrics: auc=0.7500" in capsys.readouterr().out


def test_metric_worker_factories_bind_real_fluxway_workers() -> None:
    profile = ResourceProfile(base={"CPU": 1}, ray_component_limits={"CPU": 1})
    train = get_train_metrics_aggregator_spmd_class(
        input_sharding={},
        runtime_env={},
        resource_profile=profile,
        max_inflight_steps=3,
    )
    evaluation = get_metrics_aggregator_spmd_class(
        input_sharding={},
        runtime_env={},
        resource_profile=profile,
    )

    assert train.__module__ == TrainMetricsAggregatorSPMD.__module__
    assert evaluation.__module__ == MetricsAggregatorSPMD.__module__
    assert train.__name__ == TrainMetricsAggregatorSPMD.__name__
    assert evaluation.__name__ == MetricsAggregatorSPMD.__name__
    assert train.step._fluxway_operator_spec.async_policy.compile().max_async_lag == 3


def test_metrics_component_namespace_has_no_fluxway_worker_aliases() -> None:
    metrics = importlib.import_module("flare.training.components.metrics")

    assert not hasattr(metrics, "TrainMetricsAggregatorSPMD")
    assert not hasattr(metrics, "MetricsAggregatorSPMD")
    assert not hasattr(metrics, "get_train_metrics_aggregator_spmd_class")
    assert not hasattr(metrics, "RayDataMetricsComputer")
    assert (
        importlib.util.find_spec("flare.training.components.metrics.distributed")
        is None
    )


def test_train_metric_worker_submits_component_window() -> None:
    worker = object.__new__(TrainMetricsAggregatorSPMD)
    worker._computer = Mock()
    worker._computer.write_metrics.return_value = {"global_step": 1}
    worker._prediction_refs = []
    worker._loss_sum = 0.0
    worker._sampled_step_count = 0
    worker._step_count = 1
    worker._trained_sample_count = 8
    worker._global_step = 1
    worker._window_start_time = time.time() - 0.01

    async def submit() -> None:
        await worker._submit_window()
        await worker.wait_for_metrics()

    asyncio.run(submit())

    window = worker._computer.write_metrics.call_args.args[0]
    assert isinstance(window, MetricWindow)
    assert window.step_count == 1
    assert window.trained_sample_count == 8
    assert window.prediction_refs == ()


def test_train_metric_worker_constructor_uses_ray_data_component(
    tmp_path: Path,
) -> None:
    worker = object.__new__(TrainMetricsAggregatorSPMD)
    worker.ray_component_limits = {"CPU": 1}

    TrainMetricsAggregatorSPMD.__init__(
        worker,
        log_dir=str(tmp_path / "tensorboard"),
        target_labels=np.asarray([[0, 1]], dtype=np.float32),
        target_names=("click",),
    )

    assert isinstance(worker._computer._metrics, RayDataMetricsComputer)
