from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pytest

from flare.data.contracts.schema import SchemaContract
from flare.data.contracts.transform import TransformContext
from flare.data.pipeline.production.processing.registry import TransformRegistry


class _Transform:
    @property
    def schema_contract(self) -> SchemaContract:
        schema = pa.schema([])
        return SchemaContract(schema, schema)

    def open(self, context: TransformContext) -> None:
        return None

    def apply(self, batch: pa.Table) -> pa.Table:
        return batch

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class _Factory:
    transform_id: str

    def create(self) -> _Transform:
        return _Transform()


def test_registry_resolves_explicit_factory() -> None:
    factory = _Factory("feature.encode")
    registry = TransformRegistry([factory])
    assert registry.transform_ids == ("feature.encode",)
    assert registry.require("feature.encode") is factory


def test_registry_rejects_empty_duplicate_and_unknown_factories() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TransformRegistry([])
    with pytest.raises(ValueError, match="factory id"):
        TransformRegistry([_Factory("")])
    with pytest.raises(ValueError, match="duplicate"):
        TransformRegistry([_Factory("a"), _Factory("a")])
    registry = TransformRegistry([_Factory("a")])
    with pytest.raises(KeyError, match="unknown transform"):
        registry.require("missing")
