from ..encoding.base import EncoderExpr
from ..encoding.sparse_id import SharedEmbEncoderExpr
from .base import EncoderArguments, EncoderBuildContext, FeatureGroup, FeatureSection


class SharedEmbeddingSection(FeatureSection):
    group = FeatureGroup.SHARED_EMBEDDING

    def collect(self, model_config: dict) -> dict[str, dict]:
        return {
            name: {"target_name": target}
            for name, target in model_config.get("shared_embedding_dict", {}).items()
        }

    def build_encoder(
        self,
        context: EncoderBuildContext,
        arguments: EncoderArguments,
    ) -> EncoderExpr:
        target_name = arguments["target_name"]
        if not isinstance(target_name, str) or not target_name:
            raise ValueError("shared embedding target_name must be a non-empty string")
        return SharedEmbEncoderExpr(target_slot=context.resolve_name(target_name))
