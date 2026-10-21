"""FeatureSection base — sections build EncoderExpr instances for ExprPlan."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from ..encoding.base import EncoderExpr


class FeatureGroup(enum.Enum):
    AS_IT_IS = "as_it_is"
    SHARED_EMBEDDING = "shared_embedding"
    HASH_BUCKET = "hash_bucket"
    CUT_BOUNDARY = "cut_boundary"
    FIXED_WIDTH = "fixed_width"
    CATEGORY = "category"


class EncodingPriority(enum.IntEnum):
    FALLBACK = 0
    DEFAULT = 1
    EXPLICIT = 2


@dataclass(frozen=True, slots=True)
class EncoderBuildContext:
    code: int
    length: int
    dtype: str
    resolve_name: Callable[[str], int]


EncoderArguments: TypeAlias = Mapping[str, Any]


class FeatureSection(ABC):
    group: ClassVar[FeatureGroup]
    priority: ClassVar[EncodingPriority] = EncodingPriority.EXPLICIT

    def priority_for(self, arguments: Mapping[str, object]) -> EncodingPriority:
        return self.priority

    @abstractmethod
    def collect(self, model_config: dict) -> dict[str, dict]: ...

    @abstractmethod
    def build_encoder(
        self,
        context: EncoderBuildContext,
        arguments: EncoderArguments,
    ) -> EncoderExpr: ...
