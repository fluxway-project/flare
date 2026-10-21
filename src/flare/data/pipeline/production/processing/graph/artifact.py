from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pyarrow as pa


@dataclass(frozen=True, slots=True)
class CompiledTransform:
    node_id: str
    factory_id: str
    dependencies: tuple[str, ...]
    canonical_parameters: str

    def to_state(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "factory_id": self.factory_id,
            "dependencies": list(self.dependencies),
            "canonical_parameters": self.canonical_parameters,
        }


@dataclass(frozen=True, slots=True)
class CompiledPipeline:
    pipeline_id: str
    transforms: tuple[CompiledTransform, ...]
    input_schema: pa.Schema
    output_schema: pa.Schema
    digest: str

    def __post_init__(self) -> None:
        if len(self.digest) != 64:
            raise ValueError("compiled pipeline digest must be SHA-256")
        int(self.digest, 16)

    def to_bytes(self) -> bytes:
        state = self._unsigned_state()
        state["digest"] = self.digest
        return json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, value: bytes) -> CompiledPipeline:
        state = json.loads(value)
        if not isinstance(state, dict):
            raise TypeError("compiled pipeline state must be an object")
        expected_fields = {
            "pipeline_id",
            "transforms",
            "input_schema",
            "output_schema",
            "digest",
        }
        if set(state) != expected_fields:
            raise ValueError(
                "compiled pipeline fields do not match contract: "
                f"actual={sorted(state)}"
            )
        transforms = tuple(
            CompiledTransform(
                node_id=item["node_id"],
                factory_id=item["factory_id"],
                dependencies=tuple(item["dependencies"]),
                canonical_parameters=item["canonical_parameters"],
            )
            for item in state["transforms"]
        )
        artifact = cls(
            pipeline_id=state["pipeline_id"],
            transforms=transforms,
            input_schema=cls._decode_schema(state["input_schema"]),
            output_schema=cls._decode_schema(state["output_schema"]),
            digest=state["digest"],
        )
        actual_digest = artifact.compute_digest()
        if actual_digest != artifact.digest:
            raise ValueError(
                "compiled pipeline digest mismatch: "
                f"expected={artifact.digest} actual={actual_digest}"
            )
        return artifact

    def compute_digest(self) -> str:
        canonical = json.dumps(
            self._unsigned_state(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _unsigned_state(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "transforms": [transform.to_state() for transform in self.transforms],
            "input_schema": self._encode_schema(self.input_schema),
            "output_schema": self._encode_schema(self.output_schema),
        }

    @staticmethod
    def _encode_schema(schema: pa.Schema) -> str:
        return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")

    @staticmethod
    def _decode_schema(value: str) -> pa.Schema:
        if not isinstance(value, str) or not value:
            raise ValueError("compiled pipeline schema must be encoded text")
        try:
            serialized = base64.b64decode(value, validate=True)
            return pa.ipc.read_schema(pa.BufferReader(serialized))
        except Exception as exc:
            raise ValueError("compiled pipeline schema is invalid") from exc
