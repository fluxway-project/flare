from ..encoding.base import EncoderExpr
from ..encoding.sparse_id import AsItIsEncoderExpr
from .base import (
    EncoderArguments,
    EncoderBuildContext,
    EncodingPriority,
    FeatureGroup,
    FeatureSection,
)


class AsItIsSection(FeatureSection):
    group = FeatureGroup.AS_IT_IS
    priority = EncodingPriority.DEFAULT

    def collect(self, model_config: dict) -> dict[str, dict]:
        return {name: {} for name in model_config.get("as_it_is", [])}

    def build_encoder(
        self,
        context: EncoderBuildContext,
        arguments: EncoderArguments,
    ) -> EncoderExpr:
        return AsItIsEncoderExpr()
