from collections.abc import Mapping

from ..encoding.base import EncoderExpr
from ..encoding.dense import (
    AutoCategoryEncoderExpr,
    CategoryEncoderExpr,
    CategoryLookupTable,
)
from .base import (
    EncoderArguments,
    EncoderBuildContext,
    EncodingPriority,
    FeatureGroup,
    FeatureSection,
)


class CategorySection(FeatureSection):
    group = FeatureGroup.CATEGORY

    def __init__(
        self,
        category_values: Mapping[str, Mapping[int, int]],
        auto_features: frozenset[str],
    ):
        self._category_values = {
            name: dict(values) for name, values in category_values.items()
        }
        self._auto_features = frozenset(auto_features)

    def collect(self, model_config: dict) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for name in self._auto_features:
            result[name] = {"category_mode": "auto"}
        for name in self._category_values:
            result[name] = {
                "category_mode": "mapping",
                "category_mapping": self._category_values[name],
            }
        return result

    def priority_for(self, arguments: EncoderArguments) -> EncodingPriority:
        if arguments.get("category_mode") == "auto":
            return EncodingPriority.FALLBACK
        return EncodingPriority.EXPLICIT

    def build_encoder(
        self,
        context: EncoderBuildContext,
        arguments: EncoderArguments,
    ) -> EncoderExpr:
        category_mode = arguments["category_mode"]
        if category_mode == "auto":
            return AutoCategoryEncoderExpr()
        if category_mode != "mapping":
            raise TypeError(f"unsupported category mode: {category_mode!r}")
        mapping = arguments.get("category_mapping")
        if not isinstance(mapping, Mapping):
            raise TypeError("category mapping must be provided by the Recipe")
        return CategoryEncoderExpr(table=CategoryLookupTable.from_mapping(mapping))
