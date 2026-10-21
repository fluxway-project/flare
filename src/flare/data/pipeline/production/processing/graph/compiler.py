from __future__ import annotations

import json

import pyarrow as pa

from flare.data.pipeline.production.processing.graph.artifact import (
    CompiledPipeline,
    CompiledTransform,
)
from flare.data.pipeline.production.processing.graph.logical import LogicalPipeline


class PipelineCompiler:
    def compile(
        self,
        pipeline: LogicalPipeline,
        *,
        input_schema: pa.Schema,
        output_schema: pa.Schema,
    ) -> CompiledPipeline:
        by_id = {transform.node_id: transform for transform in pipeline.transforms}
        missing = {
            dependency
            for transform in pipeline.transforms
            for dependency in transform.dependencies
            if dependency not in by_id
        }
        if missing:
            raise ValueError(
                f"logical pipeline dependencies are missing: {sorted(missing)}"
            )

        ordered: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError(f"logical pipeline contains a cycle at {node_id}")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in sorted(by_id[node_id].dependencies):
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)
            ordered.append(node_id)

        for node_id in sorted(by_id):
            visit(node_id)

        transforms = tuple(
            CompiledTransform(
                node_id=node_id,
                factory_id=by_id[node_id].factory_id,
                dependencies=tuple(sorted(by_id[node_id].dependencies)),
                canonical_parameters=json.dumps(
                    by_id[node_id].parameters,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for node_id in ordered
        )
        unsigned = CompiledPipeline(
            pipeline_id=pipeline.pipeline_id,
            transforms=transforms,
            input_schema=input_schema,
            output_schema=output_schema,
            digest="0" * 64,
        )
        return CompiledPipeline(
            pipeline_id=unsigned.pipeline_id,
            transforms=unsigned.transforms,
            input_schema=unsigned.input_schema,
            output_schema=unsigned.output_schema,
            digest=unsigned.compute_digest(),
        )
