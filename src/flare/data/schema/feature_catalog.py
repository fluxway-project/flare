from __future__ import annotations

import json
import re
import typing
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path

FEATURE_MANIFEST_FILENAME = "feature_manifest.json"
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STORAGE_COLUMN_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class FeatureRepresentation(StrEnum):
    FID = "fid"
    DENSE = "dense"


class FeatureOutputKind(StrEnum):
    SPARSE_I64_FIDS = "SPARSE_I64_FIDS"
    DENSE_I32_INDICES = "DENSE_I32_INDICES"
    DENSE_FLOAT = "DENSE_FLOAT"


def storage_column_name(code: int, representation: FeatureRepresentation) -> str:
    if isinstance(code, bool) or not isinstance(code, int) or code <= 0:
        raise ValueError(f"feature code must be a positive integer, got {code!r}")
    if not isinstance(representation, FeatureRepresentation):
        raise TypeError("representation must be a FeatureRepresentation")
    return f"{representation.value}_{code}"


def validate_storage_columns(columns: Sequence[str]) -> tuple[str, ...]:
    if not columns:
        raise ValueError("columns sequence must not be empty")
    validated: list[str] = []
    seen: set[str] = set()
    for col in columns:
        if not isinstance(col, str) or not col:
            raise ValueError(f"invalid column name: {col!r}")
        if not _STORAGE_COLUMN_PATTERN.match(col):
            raise ValueError(f"column name {col!r} violates identifier naming rules")
        if col in seen:
            raise ValueError(f"duplicate column name: {col!r}")
        seen.add(col)
        validated.append(col)
    return tuple(validated)


@dataclass(frozen=True, slots=True)
class FeatureDescriptor:
    code: int
    name: str
    representation: FeatureRepresentation
    length: int
    output_kind: FeatureOutputKind
    encoder: str
    cross_type: str | None
    cross_class: str | None
    dependency_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        storage_column_name(self.code, self.representation)
        if not self.name:
            raise ValueError("feature name must not be empty")
        if isinstance(self.length, bool) or not isinstance(self.length, int):
            raise TypeError("feature length must be an integer")
        if self.length <= 0:
            raise ValueError("feature length must be positive")
        if not isinstance(self.output_kind, FeatureOutputKind):
            raise TypeError("feature output_kind must be a FeatureOutputKind instance")
        expected_representation = (
            FeatureRepresentation.DENSE
            if self.output_kind is FeatureOutputKind.DENSE_FLOAT
            else FeatureRepresentation.FID
        )
        if self.representation is not expected_representation:
            raise ValueError(
                f"feature {self.code} representation must be "
                f"{expected_representation.value!r} for {self.output_kind.value}"
            )
        if not self.encoder:
            raise ValueError("feature encoder must not be empty")
        if self.cross_type is None and self.cross_class is not None:
            raise ValueError("cross_class requires cross_type")
        if len(self.dependency_codes) != len(set(self.dependency_codes)):
            raise ValueError(f"feature {self.code} contains duplicate dependencies")
        for dependency_code in self.dependency_codes:
            storage_column_name(dependency_code, FeatureRepresentation.FID)
        if self.code in self.dependency_codes:
            raise ValueError(f"feature {self.code} cannot depend on itself")

    @property
    def output_column(self) -> str:
        return storage_column_name(self.code, self.representation)

    def to_manifest_entry(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "output_column": self.output_column,
            "representation": self.representation.value,
            "length": self.length,
            "output_kind": self.output_kind.value,
            "encoder": self.encoder,
            "cross_type": self.cross_type,
            "cross_class": self.cross_class,
            "dependency_codes": list(self.dependency_codes),
        }


_FEATURE_ENTRY_FIELDS = (
    frozenset(field.name for field in fields(FeatureDescriptor)) - {"code"}
) | {"output_column"}


def _descriptor_from_manifest(
    code_text: str, entry: dict[str, typing.Any]
) -> FeatureDescriptor:
    if not isinstance(entry, dict):
        raise TypeError("feature manifest entry must be an object")
    if entry.keys() != _FEATURE_ENTRY_FIELDS:
        raise ValueError("feature manifest entry fields do not match")
    try:
        code = int(code_text)
    except ValueError as exc:
        raise ValueError(f"invalid feature code in manifest: {code_text!r}") from exc
    descriptor = FeatureDescriptor(
        code=code,
        name=entry["name"],
        representation=FeatureRepresentation(entry["representation"]),
        length=entry["length"],
        output_kind=FeatureOutputKind(entry["output_kind"]),
        encoder=entry["encoder"],
        cross_type=entry["cross_type"],
        cross_class=entry["cross_class"],
        dependency_codes=tuple(entry["dependency_codes"]),
    )
    if entry["output_column"] != descriptor.output_column:
        raise ValueError(
            f"feature {code} output_column must be {descriptor.output_column!r}, "
            f"got {entry['output_column']!r}"
        )
    return descriptor


class FeatureCatalog:
    def __init__(
        self,
        descriptors: Iterable[FeatureDescriptor],
        *,
        config_digest: str,
        columns: Sequence[str],
    ) -> None:
        if not _DIGEST_PATTERN.fullmatch(config_digest):
            raise ValueError("config_digest must be a lowercase SHA-256 digest")
        by_code: dict[int, FeatureDescriptor] = {}
        codes_by_name: dict[str, list[int]] = defaultdict(list)
        for descriptor in descriptors:
            if not isinstance(descriptor, FeatureDescriptor):
                raise TypeError("descriptors must contain FeatureDescriptor values")
            if descriptor.code in by_code:
                raise ValueError(f"duplicate feature code {descriptor.code}")
            by_code[descriptor.code] = descriptor
            codes_by_name[descriptor.name].append(descriptor.code)
        self._by_code = by_code
        self._codes_by_name = {
            name: tuple(sorted(codes)) for name, codes in codes_by_name.items()
        }
        self._config_digest = config_digest
        self._columns = validate_storage_columns(columns)
        missing = sorted(
            descriptor.output_column
            for descriptor in self.descriptors
            if descriptor.output_column not in self._columns
        )
        if missing:
            raise ValueError(f"feature columns missing from schema: {missing}")

    @property
    def config_digest(self) -> str:
        return self._config_digest

    @property
    def descriptors(self) -> tuple[FeatureDescriptor, ...]:
        return tuple(self._by_code[code] for code in sorted(self._by_code))

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    def require(self, code: int) -> FeatureDescriptor:
        try:
            return self._by_code[code]
        except KeyError as exc:
            raise ValueError(f"feature code {code} is not declared") from exc

    def resolve_unique_name(self, name: str) -> int:
        codes = self._codes_by_name.get(name, ())
        if not codes:
            raise ValueError(f"feature name {name!r} is not declared")
        if len(codes) != 1:
            raise ValueError(f"ambiguous feature name {name!r}: codes={list(codes)}")
        return codes[0]

    def column_for_unique_name(
        self, name: str, system_columns: Mapping[str, FeatureOutputKind] | None = None
    ) -> str:
        if system_columns and name in system_columns:
            if name not in self._columns:
                raise ValueError(f"system column {name!r} is not declared")
            return name
        return self.require(self.resolve_unique_name(name)).output_column

    def output_kind_for_unique_name(
        self, name: str, system_columns: Mapping[str, FeatureOutputKind] | None = None
    ) -> FeatureOutputKind:
        if system_columns and name in system_columns:
            if name not in self._columns:
                raise ValueError(f"system column {name!r} is not declared")
            return system_columns[name]
        return self.require(self.resolve_unique_name(name)).output_kind

    def to_manifest(self) -> dict[str, typing.Any]:
        return {
            "config_digest": self.config_digest,
            "columns": list(self._columns),
            "features": {
                str(descriptor.code): descriptor.to_manifest_entry()
                for descriptor in self.descriptors
            },
        }

    def write_manifest(self, manifest_dir: str | Path) -> Path:
        target_dir = Path(manifest_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / FEATURE_MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path

    @classmethod
    def from_manifest_dir(cls, manifest_dir: str | Path) -> FeatureCatalog:
        if manifest_dir is None:
            raise ValueError("feature manifest directory is required")
        manifest_path = Path(manifest_dir) / FEATURE_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"feature manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"feature manifest must be an object: {manifest_path}")
        expected_keys = {"config_digest", "columns", "features"}
        if set(payload) != expected_keys:
            raise ValueError(f"feature manifest fields do not match: {sorted(payload)}")
        if not isinstance(payload["columns"], list):
            raise TypeError("feature manifest columns must be an array")
        features = payload["features"]
        if not isinstance(features, dict):
            raise TypeError("feature manifest features must be an object")
        descriptors = [
            _descriptor_from_manifest(code_text, entry)
            for code_text, entry in features.items()
        ]
        return cls(
            descriptors,
            config_digest=payload["config_digest"],
            columns=payload["columns"],
        )
