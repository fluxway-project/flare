from ..encoding.base import EncoderExpr
from ..encoding.dense import BucketEncoderExpr, BucketPartition
from .base import EncoderArguments, EncoderBuildContext, FeatureGroup, FeatureSection


class CutBoundarySection(FeatureSection):
    group = FeatureGroup.CUT_BOUNDARY

    def collect(self, model_config: dict) -> dict[str, dict]:
        default_values = model_config.get("feature_default_value", {})
        return {
            name: {
                "boundaries": boundaries,
                "default_value": default_values.get(name),
            }
            for name, boundaries in model_config.get("feature_cut_boundary", {}).items()
        }

    def build_encoder(
        self,
        context: EncoderBuildContext,
        arguments: EncoderArguments,
    ) -> EncoderExpr:
        partition = BucketPartition.from_bounds(arguments["boundaries"])
        return BucketEncoderExpr(
            partition=partition,
            in_dtype=context.dtype,
            default_value=arguments["default_value"],
        )
