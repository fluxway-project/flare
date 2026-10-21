from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.types.tensor import TransportTensor

from flare.runtime.diagnostics.execution_trace import (
    ExecutionTraceOptions,
    JsonlExecutionTracer,
    create_execution_tracer,
)
from flare.runtime.diagnostics import execution_trace as execution_trace_module
from recipes.recsys.ranking.metrics.execution_trace import RecsysExecutionTracer


def _record(root: Path, stage: str, tracer: JsonlExecutionTracer) -> dict:
    tracer.flush()
    paths = tuple(root.glob(f"{stage}.*.jsonl"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text().strip())


def test_execution_trace_options_control_tracer_construction(tmp_path: Path) -> None:
    assert create_execution_tracer(ExecutionTraceOptions()) is None
    tracer = create_execution_tracer(ExecutionTraceOptions(str(tmp_path), True))
    assert isinstance(tracer, JsonlExecutionTracer)
    assert tracer.include_values is True


def test_execution_trace_options_reject_empty_enabled_directory() -> None:
    with pytest.raises(ValueError, match="directory must not be empty"):
        ExecutionTraceOptions("")


def test_generic_execution_tracer_records_event(tmp_path: Path) -> None:
    tracer = JsonlExecutionTracer(str(tmp_path))
    tracer.record_event(
        stage="forward_step",
        step=1,
        spmd_id="GenericOperator",
        rank=0,
        group="loss",
        payload={"loss": 0.42},
    )
    record = _record(tmp_path, "forward_step", tracer)
    assert record["stage"] == "forward_step"
    assert record["step"] == 1
    assert record["spmd_id"] == "GenericOperator"
    assert record["rank"] == 0
    assert record["payload"]["loss"] == 0.42


def test_trace_file_io_runs_on_background_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered_writer = threading.Event()
    release_writer = threading.Event()
    write_batch = execution_trace_module._AsyncJsonlWriter._write_batch

    def blocked_write(writer, batch) -> None:
        entered_writer.set()
        if not release_writer.wait(timeout=10):
            raise TimeoutError("test trace writer was not released")
        write_batch(writer, batch)

    monkeypatch.setattr(
        execution_trace_module._AsyncJsonlWriter, "_write_batch", blocked_write
    )
    tracer = JsonlExecutionTracer(str(tmp_path))
    tracer.record_event("forward_step", step=1, spmd_id="GenericOperator", rank=0)
    assert entered_writer.wait(timeout=5)
    assert list(tmp_path.iterdir()) == []
    release_writer.set()
    assert _record(tmp_path, "forward_step", tracer)["step"] == 1


def test_metadata_trace_does_not_read_device_values(tmp_path: Path) -> None:
    class _DeviceValue:
        shape = (2, 3)
        dtype = np.dtype("float32")

        def get(self):
            raise AssertionError("metadata trace must not copy values to host")

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("metadata trace must not coerce device values")

    value = _DeviceValue()
    tensor = SimpleNamespace(
        flat_values=value,
        global_shape=(2, 3),
        origin_indices=None,
        nested_splits=(),
    )
    tracer = RecsysExecutionTracer(str(tmp_path))
    tracer.sparse_lookup(
        group="group_dim_3",
        step=0,
        spmd_id="Embedding_group",
        rank=0,
        keys=value,
        embeddings=value,
        input_tensor=tensor,
    )
    payload = _record(tmp_path, "sparse_lookup", tracer)["payload"]
    assert payload["keys"] == {
        "shape": [2, 3],
        "dtype": "float32",
        "size": 6,
    }
    assert payload["emb"]["size"] == 6


def test_generic_execution_tracer_requires_stable_writer_identity(
    tmp_path: Path,
) -> None:
    tracer = JsonlExecutionTracer(str(tmp_path))
    with pytest.raises(ValueError, match="non-empty spmd_id"):
        tracer.record_event(
            stage="test_stage",
            step=0,
            spmd_id="",
            rank=0,
        )


def test_numeric_summary_records_non_finite_counts(tmp_path: Path) -> None:
    tracer = RecsysExecutionTracer(str(tmp_path), include_values=True)
    tracer.sparse_lookup(
        group="table_dim_2",
        step=3,
        spmd_id="Embedding_table",
        rank=0,
        keys=np.asarray([11, 13], dtype=np.int64),
        embeddings=np.asarray([[np.nan, np.inf], [-np.inf, 2.0]], dtype=np.float32),
        input_tensor=TransportTensor(
            np.asarray([11, 13], dtype=np.int64),
            global_shape=(2,),
            origin_indices=ExplicitIndex(np.asarray([0, 1], dtype=np.int64)),
        ),
    )
    record = _record(tmp_path, "sparse_lookup", tracer)
    summary = record["payload"]["emb"]
    assert summary["shape"] == [2, 2]
    assert summary["dtype"] == "float32"
    assert summary["size"] == 4


def test_sparse_precision_warning_reports_without_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tracer = RecsysExecutionTracer(str(tmp_path))
    with caplog.at_level("WARNING"):
        tracer.report_sparse_nonfinite(
            group="table_dim_2",
            step=3,
            spmd_id="Embedding_table",
            rank=0,
            boundary="model_gather",
            values=np.asarray([[np.nan, np.inf]], dtype=np.float16),
        )
    assert "non-finite sparse FP16 values" in caplog.text
    record = _record(tmp_path, "sparse_precision_warning", tracer)
    assert record["payload"]["boundary"] == "model_gather"
    assert record["payload"]["values"]["nan_count"] == 1
    assert record["payload"]["values"]["inf_count"] == 1


def test_module_update_records_gradient_owner_fingerprints(tmp_path: Path) -> None:
    tracer = RecsysExecutionTracer(str(tmp_path))
    raw = {"sum": float("nan"), "fingerprint": "raw"}
    reduced = {"sum": float("nan"), "fingerprint": "reduced"}
    accumulated = {"sum": float("nan"), "fingerprint": "accumulated"}
    tracer.module_update(
        step=0,
        spmd_id="RecsysBackboneTrain",
        rank=0,
        micro_step=1,
        grad_accum_steps=3,
        optimizer_applied=False,
        module_before={"fingerprint": "model"},
        module_after={"fingerprint": "model"},
        raw_gradients=raw,
        reduced_gradients=reduced,
        accumulated_gradients=accumulated,
    )
    payload = _record(tmp_path, "model_update", tracer)["payload"]
    assert payload["raw_gradients"]["fingerprint"] == "raw"
    assert np.isnan(payload["raw_gradients"]["sum"])
    assert payload["reduced_gradients"]["fingerprint"] == "reduced"
    assert np.isnan(payload["reduced_gradients"]["sum"])
    assert payload["accumulated_gradients"]["fingerprint"] == "accumulated"
    assert np.isnan(payload["accumulated_gradients"]["sum"])


def test_state_fingerprint_handles_local_shards_and_non_finite_values() -> None:
    probe = Path(__file__).with_name("state_audit_runtime_probe.py")
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "STATE_AUDIT_RUNTIME=PASS" in result.stdout


def test_recsys_execution_tracer_records_exact_sparse_and_module_values(
    tmp_path: Path,
) -> None:
    tracer = RecsysExecutionTracer(str(tmp_path), include_values=True)
    origins = ExplicitIndex(np.asarray([5, 9], dtype=np.int64))
    ids = TransportTensor(
        np.asarray([11, 13], dtype=np.int64),
        global_shape=(2,),
        origin_indices=origins,
        source_step=3,
    )
    inverse = TransportTensor(
        np.asarray([0, 1], dtype=np.int32),
        nested_splits=[np.asarray([0, 2], dtype=np.int32)],
        global_shape=(2, 1),
        source_step=3,
    )
    embeddings = np.asarray([[1.25, -2.5], [3.75, -4.0]], dtype=np.float32)
    sparse_input = np.asarray([[[1.25, -2.5], [3.75, -4.0]]], dtype=np.float32)

    tracer.input_unique(
        group="table_dim_2",
        step=3,
        spmd_id="Input",
        rank=0,
        local_rows=2,
        original_ids=ids.flat_values,
        unique_ids=ids.flat_values,
        inverse=inverse.flat_values,
        unique_global_positions=np.asarray([5, 9], dtype=np.int64),
        unique_tensor=ids,
        inverse_tensor=inverse,
        world_size=2,
    )
    tracer.sparse_lookup(
        group="table_dim_2",
        step=3,
        spmd_id="Embedding_table",
        rank=0,
        keys=ids.flat_values,
        embeddings=embeddings,
        input_tensor=ids,
    )
    tracer.module_gather(
        group="table_dim_2",
        step=3,
        spmd_id="Model",
        rank=0,
        embeddings=embeddings,
        inverse=inverse.flat_values,
        sparse_input=sparse_input,
        embedding_tensor=TransportTensor(embeddings, global_shape=(2, 2)),
        inverse_tensor=inverse,
    )

    input_record = _record(tmp_path, "input_unique", tracer)
    assert input_record["payload"]["unique_global_positions"] == [5, 9]
    assert input_record["payload"]["nested_splits_values"] == [[0, 2]]
    lookup_record = _record(tmp_path, "sparse_lookup", tracer)
    assert lookup_record["payload"]["embedding_values"] == embeddings.tolist()
    assert lookup_record["payload"]["origin_values"] == [5, 9]
    gather_record = _record(tmp_path, "model_gather", tracer)
    assert gather_record["payload"]["sparse_input_values"] == sparse_input.tolist()


def test_recsys_execution_tracer_handles_none_nested_splits(tmp_path: Path) -> None:
    tracer = RecsysExecutionTracer(str(tmp_path), include_values=True)
    ids = TransportTensor(
        np.asarray([10], dtype=np.int64),
        global_shape=(1,),
        origin_indices=ExplicitIndex(np.asarray([0], dtype=np.int64)),
        source_step=1,
    )
    inverse = TransportTensor(
        np.asarray([0], dtype=np.int32),
        global_shape=(1,),
        source_step=1,
    )
    tracer.input_unique(
        group="group1",
        step=1,
        spmd_id="Input",
        rank=0,
        local_rows=1,
        original_ids=ids.flat_values,
        unique_ids=ids.flat_values,
        inverse=inverse.flat_values,
        unique_global_positions=np.asarray([0], dtype=np.int64),
        unique_tensor=ids,
        inverse_tensor=inverse,
        world_size=1,
    )
    input_record = _record(tmp_path, "input_unique", tracer)
    assert input_record["payload"]["nested_splits_values"] == []
