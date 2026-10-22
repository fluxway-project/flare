from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def ranking_forward(model: Any, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    if not callable(model):
        raise TypeError(f"ranking model must be callable, got {type(model).__name__}")
    if not isinstance(inputs, Mapping):
        raise TypeError(
            f"ranking inputs must be a Mapping, got {type(inputs).__name__}"
        )
    outputs = model(inputs)
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"ranking model output must be a Mapping, got {type(outputs).__name__}"
        )
    if "logits" not in outputs:
        raise KeyError("ranking model output requires 'logits'")
    return outputs


__all__ = ["ranking_forward"]
