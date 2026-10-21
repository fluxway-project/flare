from __future__ import annotations

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import tyro
from fluxway.common.types.tensor import TransportTensor

from recipes.recsys.ranking.benchmark.throughput_sink import ThroughputSinkSPMD
from recipes.recsys.ranking.workers.reader import resolve_reader_async_policy
from recipes.whw.util.config import ModelEnum, ThroughputBenchmarkConfig
from recipes.whw.util.throughput_metrics import summarize_sink_reports


def make_config(**overrides):
    values = {
        "conf_version": ModelEnum.conf_geek_predict_gpu_mix_op_s2_test,
        "reader_workers": 24,
        "dp": 8,
        "samples_per_input_step": 6144,
        "steps": 4,
        "storage_format": "iceberg",
    }
    values.update(overrides)
    return ThroughputBenchmarkConfig(**values)


def make_reports():
    return [
        {
            "rank": rank,
            "world_size": 8,
            "graph_steps": 3,
            "step_valid_samples": [6144] * 3,
            "valid_samples": 6144 * 3,
            "step_intervals": [2.0, 3.0],
            "measured_duration_seconds": 5.0,
        }
        for rank in range(8)
    ]


def summarize(reports):
    return summarize_sink_reports(
        reports,
        input_workers=8,
        reader_workers=24,
        samples_per_input_step=6144,
        global_batch_size=49152,
        steps=3,
        wall_seconds=10.0,
    )


def test_batch_is_derived_and_cli_rejects_global_override():
    config = make_config()
    assert config.batch_size == 49152
    with pytest.raises(TypeError):
        make_config(batch_size=6144)
    arguments = [
        "--conf-version",
        config.conf_version.name,
        "--reader-workers",
        "24",
        "--dp",
        "8",
        "--samples-per-input-step",
        "6144",
        "--steps",
        "4",
        "--storage-format",
        "iceberg",
    ]
    assert tyro.cli(ThroughputBenchmarkConfig, args=arguments).batch_size == 49152
    with pytest.raises(SystemExit) as caught:
        tyro.cli(ThroughputBenchmarkConfig, args=[*arguments, "--batch-size", "6144"])
    assert caught.value.code != 0
    with pytest.raises(SystemExit) as caught:
        tyro.cli(
            ThroughputBenchmarkConfig,
            args=[*arguments, "--reader-unit-size", "6144"],
        )
    assert caught.value.code != 0


def test_reader_pipeline_depth_maps_to_exact_policy():
    sync = resolve_reader_async_policy(1).compile()
    assert sync.mode == "sync"
    assert sync.pipeline_depth == 1
    assert sync.max_inflight_steps == 0
    assert sync.state_access == "exclusive"

    pipelined = resolve_reader_async_policy(4).compile()
    assert pipelined.mode == "pipelined"
    assert pipelined.pipeline_depth == 4
    assert pipelined.max_inflight_steps == 3
    assert pipelined.state_access == "exclusive"


@pytest.mark.parametrize("depth", [0, -1, True, 1.5])
def test_reader_pipeline_depth_rejects_invalid_values(depth):
    with pytest.raises(ValueError, match="reader_pipeline_depth"):
        resolve_reader_async_policy(depth)


@pytest.mark.parametrize(
    "overrides",
    [
        {"samples_per_input_step": 0},
        {"samples_per_input_step": True},
        {"samples_per_input_step": 1.5},
        {"dp": 0},
        {"reader_workers": 25},
        {"steps": None},
        {"steps": 1},
        {"steps": True},
        {"jax_platform": "cuda"},
    ],
)
def test_invalid_benchmark_parameters_fail(overrides):
    with pytest.raises(ValueError):
        make_config(**overrides)


def test_metrics_count_only_samples_inside_arrival_window():
    result = summarize(make_reports())
    assert result["valid_samples"] == 8 * 6144 * 3
    assert result["arrival_window_samples"] == 8 * 6144 * 2
    assert result["arrival_window_aggregate_samples_per_second"] == 8 * 6144 * 2 / 5
    assert result["samples_per_input_step"] == 6144
    assert result["global_batch_size"] == 49152
    assert result["reader_interleave_factor"] == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", False),
        ("rank", 1),
        ("world_size", 7),
        ("graph_steps", 2),
        ("step_valid_samples", [768] * 3),
        ("step_valid_samples", [6144, 6144, 0]),
        ("step_intervals", [2.0]),
        ("step_intervals", [2.0, float("inf")]),
        ("valid_samples", 1),
        ("measured_duration_seconds", float("nan")),
    ],
)
def test_wrong_reports_are_not_performance_results(field, value):
    reports = copy.deepcopy(make_reports())
    reports[0][field] = value
    with pytest.raises(ValueError):
        summarize(reports)


def test_missing_rank_fails():
    with pytest.raises(ValueError):
        summarize(make_reports()[:-1])


def make_sink(path, steps):
    sink = object.__new__(ThroughputSinkSPMD)
    ThroughputSinkSPMD.__init__(sink, max_steps=steps, report_output_path=str(path))
    sink.rank = 0
    sink.world_size = 1
    return sink


@pytest.mark.parametrize("steps", [1, 3])
def test_sink_consume_and_report_use_same_window(tmp_path, steps):
    sink = make_sink(tmp_path / "report.json", steps)
    count = TransportTensor.from_tensor(
        np.asarray([6144], dtype=np.int32), global_shape=(1,)
    )
    with patch(
        "recipes.recsys.ranking.benchmark.throughput_sink.time.perf_counter",
        side_effect=[10.0, 12.0, 15.0],
    ):
        for _ in range(steps):
            sink.consume(samples_valid_count=count)
    report = json.loads((tmp_path / "report_rank_0.json").read_text())
    assert report["arrival_window_valid_samples"] == (steps - 1) * 6144
    assert report["arrival_window_per_input_samples_per_second"] == (
        None if steps == 1 else 12288 / 5
    )
    assert "local_samples_per_second" not in report


def test_sink_rejects_invalid_samples_before_recording(tmp_path):
    sink = make_sink(tmp_path / "report.json", 3)
    count = TransportTensor.from_tensor(
        np.asarray([0], dtype=np.int32), global_shape=(1,)
    )
    with pytest.raises(ValueError, match="positive"):
        sink.consume(samples_valid_count=count)
    with pytest.raises(KeyError):
        sink.consume()
    assert sink._current_step == 0
    assert sink._step_timestamps == []
