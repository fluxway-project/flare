from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pytest

from flare.data.contracts.schema import SchemaContract
from flare.data.contracts.sink import (
    PublicationIdentity,
    SinkResult,
    StreamingSink,
    TransactionalSink,
)
from flare.data.contracts.transform import (
    BatchTransform,
    TransformContext,
    TransformFactory,
)


class _Transform:
    def __init__(self, contract: SchemaContract) -> None:
        self._contract = contract
        self.events: list[str] = []

    @property
    def schema_contract(self) -> SchemaContract:
        return self._contract

    def open(self, context: TransformContext) -> None:
        self.events.append(f"open:{context.worker_id}")

    def apply(self, batch: pa.Table) -> pa.Table:
        self._contract.validate_input(batch)
        result = pa.table({"output": batch["input"]})
        self._contract.validate_output(result)
        self.events.append("apply")
        return result

    def close(self) -> None:
        self.events.append("close")


@dataclass(frozen=True)
class _Factory:
    contract: SchemaContract

    @property
    def transform_id(self) -> str:
        return "test.transform-v1"

    def create(self) -> _Transform:
        return _Transform(self.contract)


class _Transactional:
    def publish(
        self,
        payload: object,
        *,
        schema: pa.Schema,
        publication: PublicationIdentity,
    ) -> SinkResult:
        return SinkResult(total_rows=1)


class _Streaming:
    def consume(self, payload: object, *, schema: pa.Schema) -> SinkResult:
        return SinkResult(total_rows=1)


def test_transform_lifecycle_and_schema_contract() -> None:
    contract = SchemaContract(
        input_schema=pa.schema([pa.field("input", pa.int64())]),
        output_schema=pa.schema([pa.field("output", pa.int64())]),
    )
    factory = _Factory(contract)
    transform = factory.create()
    context = TransformContext(stage_id="stage", worker_id="worker", resources={})

    assert isinstance(factory, TransformFactory)
    assert isinstance(transform, BatchTransform)
    transform.open(context)
    result = transform.apply(pa.table({"input": pa.array([1], pa.int64())}))
    transform.close()

    assert result.to_pydict() == {"output": [1]}
    assert transform.events == ["open:worker", "apply", "close"]


def test_schema_contract_rejects_mismatch() -> None:
    contract = SchemaContract(
        input_schema=pa.schema([pa.field("input", pa.int64())]),
        output_schema=pa.schema([pa.field("output", pa.int64())]),
    )
    with pytest.raises(ValueError, match="input batch schema mismatch"):
        contract.validate_input(pa.table({"input": pa.array([1], pa.int32())}))


def test_sink_contracts_are_mutually_explicit() -> None:
    transactional = _Transactional()
    streaming = _Streaming()
    assert isinstance(transactional, TransactionalSink)
    assert not isinstance(transactional, StreamingSink)
    assert isinstance(streaming, StreamingSink)
    assert not isinstance(streaming, TransactionalSink)


def test_contract_values_reject_invalid_state() -> None:
    with pytest.raises(ValueError, match="publication identity"):
        PublicationIdentity("")
    with pytest.raises(ValueError, match="total_rows"):
        SinkResult(-1)
    with pytest.raises(ValueError, match="stage_id"):
        TransformContext(stage_id="", worker_id="worker", resources={})
    with pytest.raises(ValueError, match="resources"):
        TransformContext(stage_id="stage", worker_id="worker", resources={"CPU": -1})
