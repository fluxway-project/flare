from ..encoding.base import EncoderExpr
from ..encoding.dense import FixedWidthBucketEncoderExpr
from .base import EncoderArguments, EncoderBuildContext, FeatureGroup, FeatureSection


class FixedWidthSection(FeatureSection):
    group = FeatureGroup.FIXED_WIDTH

    def collect(self, model_config: dict) -> dict[str, dict]:
        default_values = model_config.get("feature_default_value", {})
        return {
            name: {
                "width": width,
                "default_value": default_values.get(name),
            }
            for name, width in model_config.get(
                "feature_cut_by_fixed_width", {}
            ).items()
        }

    def build_encoder(
        self,
        context: EncoderBuildContext,
        arguments: EncoderArguments,
    ) -> EncoderExpr:
        return FixedWidthBucketEncoderExpr(
            width=float(arguments["width"]),
            in_dtype=context.dtype,
            default_value=arguments["default_value"],
        )
