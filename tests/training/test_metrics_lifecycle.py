from __future__ import annotations

import asyncio
import inspect
from unittest.mock import Mock

import numpy as np
import pytest

from flare.runtime.fluxway.transport import make_transport_tensor
from flare.training.components.metrics.contracts import MetricSink
from recipes.recsys.ranking.metrics import eval_metrics, train_metrics


class ControlledMetrics:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.error: RuntimeError | None = None
        self.finished_rows: list[int] = []
        self.tasks: list[asyncio.Task[dict[str, dict[str, object]]]] = []

    def put_batch(self, *, scores, labels, groups):
        assert scores.shape[0] == labels.shape[0] == groups.shape[0]
        return labels.shape[0]

    def submit_compute(self, refs):
        # Ray starts remote work at submission, before ObjectRef is awaited.
        task = asyncio.create_task(self._compute(refs))
        self.tasks.append(task)
        return task

    async def _compute(self, refs):
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        rows = sum(refs)
        self.finished_rows.append(rows)
        return {
            "click": {
                "sample_count": rows,
                "auc": 0.75,
                "auc_defined": True,
                "grouped_auc": 0.75,
                "grouped_auc_defined": True,
            }
        }


@pytest.fixture(params=("train", "eval"))
def metric_worker(request, monkeypatch):
    computer = ControlledMetrics()
    sink = Mock(spec=MetricSink)
    module = train_metrics if request.param == "train" else eval_metrics
    monkeypatch.setattr(module, "RayDataMetricsComputer", lambda **_: computer)
    if request.param == "train":
        monkeypatch.setattr(module, "RankingTensorBoardMetricSink", lambda _: sink)
    else:
        import ray

        async def await_result(ref):
            return await ref

        def get_result(ref):
            return asyncio.run_coroutine_threadsafe(
                await_result(ref), ref.get_loop()
            ).result()

        monkeypatch.setattr(ray, "get", get_result)

    def create(*, window_steps: int, target_step: int | None):
        worker_class = (
            train_metrics.TrainMetricsAggregatorSPMD
            if request.param == "train"
            else eval_metrics.MetricsAggregatorSPMD
        )
        worker = object.__new__(worker_class)
        worker.ray_component_limits = {"CPU": 1}
        arguments = {
            "window_steps": window_steps,
            "target_labels": np.asarray([[0, 1]], dtype=np.float32),
            "target_names": ("click",),
        }
        if request.param == "train":
            arguments.update(log_dir="unused", target_step=target_step)
        else:
            arguments.update(max_steps=target_step)
        worker_class.__init__(worker, **arguments)
        return worker

    return create, computer


def metric_batch(*, valid_rows: int = 2):
    return {
        name: make_transport_tensor(value)
        for name, value in {
            "scores": np.asarray([[0.2], [0.8]], dtype=np.float32),
            "labels": np.asarray([0, 1], dtype=np.int32),
            "group_ids": np.asarray([0, 0], dtype=np.int64),
            "loss": np.asarray([1.0], dtype=np.float32),
            "metric_sampled": np.asarray([1], dtype=np.int32),
            "samples_valid_count": np.asarray([valid_rows], dtype=np.int32),
        }.items()
    }


async def invoke_step(worker, **batch):
    result = worker.step(**batch)
    if inspect.isawaitable(result):
        return await result
    return result


def test_metric_step_cannot_complete_before_its_reduction(metric_worker):
    create, computer = metric_worker
    worker = create(window_steps=1, target_step=1)

    async def scenario():
        computer.release.clear()
        step = asyncio.create_task(invoke_step(worker, **metric_batch()))
        try:
            await asyncio.wait_for(computer.started.wait(), timeout=1)
            assert not step.done(), "metric work escaped operator completion"
            computer.release.set()
            await step
            assert computer.finished_rows == [2]
        finally:
            computer.release.set()
            await asyncio.gather(step, worker.on_shutdown(), return_exceptions=True)

    asyncio.run(scenario())


def test_metric_reduction_error_fails_its_step(metric_worker):
    create, computer = metric_worker
    worker = create(window_steps=1, target_step=1)
    computer.error = RuntimeError("metric reduction failed")

    async def scenario():
        try:
            with pytest.raises(RuntimeError, match="metric reduction failed"):
                await invoke_step(worker, **metric_batch())
        finally:
            await asyncio.gather(worker.on_shutdown(), return_exceptions=True)

    asyncio.run(scenario())


def test_terminal_partial_window_finishes_before_terminal_step(metric_worker):
    create, computer = metric_worker
    worker = create(window_steps=2, target_step=3)

    async def scenario():
        try:
            await invoke_step(worker, **metric_batch())
            await invoke_step(worker, **metric_batch())
            await invoke_step(worker, **metric_batch(valid_rows=1))
            assert computer.finished_rows == [4, 1]
        finally:
            await worker.on_shutdown()
        assert computer.finished_rows == [4, 1]

    asyncio.run(scenario())


def test_source_exhaustion_flushes_only_remaining_valid_rows(metric_worker):
    create, computer = metric_worker
    worker = create(window_steps=4, target_step=None)

    async def scenario():
        await invoke_step(worker, **metric_batch())
        await invoke_step(worker, **metric_batch(valid_rows=1))
        await worker.on_shutdown()
        assert computer.finished_rows == [3]

    asyncio.run(scenario())
