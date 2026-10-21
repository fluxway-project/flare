from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cloudpickle

from flare.data.pipeline.production.builder.expr_plan import ExprPlan
from flare.data.pipeline.production.builder.row_parser import build_feature_processor
from flare.data.pipeline.production.processing.graph import (
    LogicalPipeline,
    LogicalTransform,
    PipelineCompiler,
)
from flare.data.pipeline.production.processing.graph.artifact import CompiledPipeline
from flare.data.pipeline.production.processing.parser_contract import ParserContract
from flare.data.pipeline.production.processing.parser_spec import (
    ParserSpec,
    build_parser_spec,
)
from flare.data.schema.feature_catalog import FeatureCatalog


@dataclass(frozen=True, slots=True)
class FeatureProcessingArtifact:
    compiled_pipeline: CompiledPipeline
    parser_spec: ParserSpec
    metadata_maps: Mapping[str, Mapping[str, int]]
    encoded_plan: bytes
    feature_catalog: FeatureCatalog

    def load_expr_plan(self) -> ExprPlan:
        value = cloudpickle.loads(self.encoded_plan)
        if not isinstance(value, ExprPlan):
            raise TypeError("processing artifact does not contain an ExprPlan")
        return value


class ProcessingArtifactCompiler:
    def compile(
        self,
        *,
        conf_dir: Path,
        model_conf_dir: Path,
        parser_contract: ParserContract,
        category_values: Mapping[str, Mapping[int, int]] | None = None,
        auto_category_features: frozenset[str] = frozenset(),
    ) -> FeatureProcessingArtifact:
        processor = build_feature_processor(
            conf_dir,
            model_conf_dir,
            metadata_columns=parser_contract.metadata_columns,
            category_values=category_values,
            auto_category_features=auto_category_features,
        )
        parser_spec = build_parser_spec(
            processor,
            metadata_columns=parser_contract.metadata_columns,
            direct_columns=parser_contract.direct_columns,
            feature_payload_column_index=parser_contract.feature_payload_column_index,
        )
        output_schema = processor.build_vortex_schema()
        plan = ExprPlan(
            fea_cross_ops=processor.build_fea_cross_ops(),
            encode_specs=processor.build_encode_specs(),
            fid_cross_ops=processor.build_fid_cross_ops(),
            output_cols=output_schema.names,
            post_fid_encode_specs=processor.build_post_fid_encode_specs(),
        )
        logical = LogicalPipeline(
            pipeline_id="flare-feature-processing",
            transforms=(
                LogicalTransform(
                    node_id="parse",
                    factory_id="flare.parse",
                    dependencies=(),
                    parameters={
                        "config_digest": processor.feature_catalog.config_digest,
                        "raw_schema": parser_spec.raw_schema.to_string(),
                    },
                ),
                LogicalTransform(
                    node_id="feature-transform",
                    factory_id="flare.feature-transform",
                    dependencies=("parse",),
                    parameters={
                        "config_digest": processor.feature_catalog.config_digest,
                        "output_columns": output_schema.names,
                    },
                ),
            ),
        )
        compiled = PipelineCompiler().compile(
            logical,
            input_schema=parser_spec.raw_schema,
            output_schema=output_schema,
        )
        encoded_plan = cloudpickle.dumps(plan)
        restored_plan = cloudpickle.loads(encoded_plan)
        if not isinstance(restored_plan, ExprPlan):
            raise TypeError("compiled expression plan round-trip changed its type")
        if (
            hashlib.sha256(encoded_plan).digest()
            != hashlib.sha256(cloudpickle.dumps(restored_plan)).digest()
        ):
            raise RuntimeError("compiled expression plan is not deterministic")
        return FeatureProcessingArtifact(
            compiled_pipeline=compiled,
            parser_spec=parser_spec,
            metadata_maps=parser_contract.metadata_maps,
            encoded_plan=encoded_plan,
            feature_catalog=processor.feature_catalog,
        )
