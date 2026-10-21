from flare.data.pipeline.production.builder.constants import DEFAULT_NULL_SENTINEL

from ..encoding.base import EncoderExpr
from ..encoding.sparse_id import HashEncoderExpr
from .base import EncoderArguments, EncoderBuildContext, FeatureGroup, FeatureSection


class HashBucketSection(FeatureSection):
    group = FeatureGroup.HASH_BUCKET

    def collect(self, model_config: dict) -> dict[str, dict]:
        default_values = model_config.get("feature_default_value", {})
        return {
            name: {
                "bucket_size": size,
                "default_value": default_values.get(name, DEFAULT_NULL_SENTINEL),
            }
            for name, size in model_config.get("hash_bucket_size", {}).items()
        }

    def build_encoder(
        self,
        context: EncoderBuildContext,
        arguments: EncoderArguments,
    ) -> EncoderExpr:
        return HashEncoderExpr(
            bucket_size=int(arguments["bucket_size"]),
            default_value=int(arguments["default_value"]),
        )
