from __future__ import annotations

import json

import pyarrow as pa
import pytest

from flare.data.pipeline.production.processing.graph.artifact import CompiledPipeline
from flare.data.pipeline.production.processing.graph.compiler import PipelineCompiler
from flare.data.pipeline.production.processing.graph.logical import (
    LogicalPipeline,
    LogicalTransform,
)


def _pipeline() -> LogicalPipeline:
    return LogicalPipeline(
        pipeline_id="training",
        transforms=(
            LogicalTransform(
                "encode", "encode", ("parse",), {"output_columns": ["fid_1"]}
            ),
            LogicalTransform("parse", "parse", (), {"format": "arc"}),
        ),
    )


def test_compiler_canonicalizes_topology_and_round_trips() -> None:
    schema = pa.schema([pa.field("value", pa.int64())])
    artifact = PipelineCompiler().compile(
        _pipeline(), input_schema=schema, output_schema=schema
    )
    restored = CompiledPipeline.from_bytes(artifact.to_bytes())

    assert tuple(item.node_id for item in artifact.transforms) == ("parse", "encode")
    assert restored == artifact
    assert restored.digest == artifact.compute_digest()


def test_compiler_digest_is_independent_of_declaration_and_mapping_order() -> None:
    schema = pa.schema([pa.field("value", pa.int64())])
    first = PipelineCompiler().compile(
        _pipeline(), input_schema=schema, output_schema=schema
    )
    reordered = LogicalPipeline(
        pipeline_id="training",
        transforms=(
            LogicalTransform("parse", "parse", (), {"format": "arc"}),
            LogicalTransform(
                "encode", "encode", ("parse",), {"output_columns": ["fid_1"]}
            ),
        ),
    )
    second = PipelineCompiler().compile(
        reordered, input_schema=schema, output_schema=schema
    )
    assert first.digest == second.digest


def test_compiler_rejects_missing_dependency_cycle_and_duplicate_node() -> None:
    schema = pa.schema([])
    compiler = PipelineCompiler()
    with pytest.raises(ValueError, match="missing"):
        compiler.compile(
            LogicalPipeline(
                "missing",
                (LogicalTransform("node", "factory", ("absent",), {}),),
            ),
            input_schema=schema,
            output_schema=schema,
        )
    with pytest.raises(ValueError, match="cycle"):
        compiler.compile(
            LogicalPipeline(
                "cycle",
                (
                    LogicalTransform("a", "factory", ("b",), {}),
                    LogicalTransform("b", "factory", ("a",), {}),
                ),
            ),
            input_schema=schema,
            output_schema=schema,
        )
    with pytest.raises(ValueError, match="unique"):
        LogicalPipeline(
            "duplicate",
            (
                LogicalTransform("a", "factory", (), {}),
                LogicalTransform("a", "factory", (), {}),
            ),
        )


def test_artifact_rejects_tampering() -> None:
    schema = pa.schema([])
    artifact = PipelineCompiler().compile(
        _pipeline(), input_schema=schema, output_schema=schema
    )
    state = json.loads(artifact.to_bytes())
    state["pipeline_id"] = "tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        CompiledPipeline.from_bytes(json.dumps(state).encode())
