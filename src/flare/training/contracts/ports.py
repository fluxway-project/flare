from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int | None, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.dtype:
            raise ValueError("TensorSpec dtype cannot be empty")


@dataclass(frozen=True)
class PortSpec:
    name: str
    spec: TensorSpec | None = None
    is_optional: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PortSpec name cannot be empty")


__all__ = ["PortSpec", "TensorSpec"]
