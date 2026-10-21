from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol


class BatchEvaluator(Protocol):
    def evaluate_batch(self, batch: Any) -> Any: ...


@dataclass(frozen=True)
class EvaluationEnvelope:
    payload: Any
    request_id: str
    schema_id: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("EvaluationEnvelope request_id must be a non-empty string")
        if not isinstance(self.schema_id, str) or not self.schema_id:
            raise ValueError("EvaluationEnvelope schema_id must be a non-empty string")
        if not isinstance(self.metadata, MappingProxyType):
            raise TypeError(
                "EvaluationEnvelope metadata must be a MappingProxyType, "
                f"got {type(self.metadata).__name__}"
            )

    @classmethod
    def from_metadata(
        cls,
        *,
        payload: Any,
        request_id: str,
        schema_id: str,
        metadata: Mapping[str, Any],
    ) -> EvaluationEnvelope:
        return cls(
            payload=payload,
            request_id=request_id,
            schema_id=schema_id,
            metadata=MappingProxyType(dict(metadata)),
        )


__all__ = ["BatchEvaluator", "EvaluationEnvelope"]
