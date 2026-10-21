from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_input_gradient_keys(
    input_gradient_keys: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if input_gradient_keys is None:
        return None
    if not isinstance(input_gradient_keys, tuple):
        raise TypeError("input_gradient_keys must be a tuple when provided")
    if any(not isinstance(key, str) or not key for key in input_gradient_keys):
        raise ValueError("input_gradient_keys must contain non-empty strings")
    if len(set(input_gradient_keys)) != len(input_gradient_keys):
        raise ValueError("input_gradient_keys must be unique")
    return input_gradient_keys


def partition_input_gradients(
    inputs: Mapping[str, Any],
    input_gradient_keys: tuple[str, ...] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[str, ...]]:
    if input_gradient_keys is None:
        return {}, inputs, tuple(inputs)
    missing = tuple(key for key in input_gradient_keys if key not in inputs)
    if missing:
        raise KeyError(f"input gradient keys are missing from inputs: {missing}")
    differentiable_inputs = {key: inputs[key] for key in input_gradient_keys}
    module_inputs = {
        key: value for key, value in inputs.items() if key not in input_gradient_keys
    }
    return module_inputs, differentiable_inputs, input_gradient_keys


__all__ = ["partition_input_gradients", "validate_input_gradient_keys"]
