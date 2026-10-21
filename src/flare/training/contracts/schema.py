from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from flare.training.contracts.ports import TensorSpec


@dataclass(frozen=True)
class BatchFieldSpec:
    name: str
    spec: TensorSpec
    is_sparse: bool = False
    is_label: bool = False
    is_weight: bool = False


@dataclass(frozen=True)
class BatchSchema:
    fields: Mapping[str, BatchFieldSpec]

    def __getitem__(self, name: str) -> BatchFieldSpec:
        return self.fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self.fields


__all__ = ["BatchFieldSpec", "BatchSchema"]
