"""FeatureProcessor — config-driven encoder & cross plan builder.

Reads ``feature.tsv`` + ``model_config.json`` + cross tsvs,
resolves cross dependency closures, instantiates the relevant
``EncoderExpr`` / ``FeaCrossExpr`` / ``FidCrossExpr`` objects, and exposes
them in a form directly consumable by
:class:`flare.data.pipeline.production.builder.expr_plan.ExprPlan` and the Vortex writer.

All feature transformation happens via the PyArrow Acero plan; this
module is now a pure metadata / factory layer.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa

from flare.data.schema.feature_catalog import (
    FeatureCatalog,
    FeatureDescriptor,
    FeatureOutputKind,
    FeatureRepresentation,
    storage_column_name,
)

from .cross.base import get_fea_factory, get_fid_factory
from .cross.fid_ops import FidCrossExpr as FidCrossNExpr
from .cross.protocol import FeaCrossExpr, FidCrossExpr
from .encoding.base import EncoderExpr
from .encoding.dense import AutoCategoryEncoderExpr, CategoryEncoderExpr
from .encoding.protocol import output_kind_pa_type
from .section import (
    AsItIsSection,
    CategorySection,
    CutBoundarySection,
    EncoderBuildContext,
    FeatureGroup,
    FeatureSection,
    FixedWidthSection,
    HashBucketSection,
    SharedEmbeddingSection,
)


@dataclass(frozen=True)
class EncodingConfigPlan:
    assignments: Mapping[str, EncodingAssignment]


@dataclass(frozen=True, slots=True)
class EncodingAssignment:
    section: FeatureSection
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    code: int
    name: str
    length: int
    dtype: str
    cross_type: str | None = None
    cross_class: str | None = None
    dependency_codes: tuple[int, ...] = ()

    @property
    def is_cross(self) -> bool:
        return self.cross_type is not None


@dataclass(frozen=True, slots=True)
class CrossDefinition:
    code: int
    name: str
    cross_type: str
    cross_class: str
    dependency_codes: tuple[int, ...]


def string_to_pa_type(dtype_str: str) -> pa.DataType:
    dtype_str = dtype_str.lower().strip()
    mapping = {
        "int64": pa.int64(),
        "int32": pa.int32(),
        "int16": pa.int16(),
        "int8": pa.int8(),
        "float32": pa.float32(),
        "float": pa.float32(),
        "float64": pa.float64(),
        "double": pa.float64(),
        "string": pa.string(),
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype in feature.tsv: {dtype_str}")
    return mapping[dtype_str]


class FeatureProcessor:
    """Builds EncoderExpr / cross-op factories from configuration files."""

    def __init__(
        self,
        feature_tsv: str,
        model_config_json: str,
        category_feature_dir: str | None = None,
        *,
        category_values: Mapping[str, Mapping[int, int]] | None = None,
        auto_category_features: Iterable[str] = (),
        model_input_feature_names: Iterable[str] | None = None,
        metadata_columns: Mapping[str, int] | None = None,
    ):
        if category_feature_dir is not None:
            raise ValueError(
                "category_feature_dir is not supported; provide Recipe-owned "
                "category_values instead"
            )
        if model_input_feature_names is None:
            raise ValueError("model_input_feature_names is required")
        model_input_names = frozenset(model_input_feature_names)
        if not model_input_names:
            raise ValueError("model_input_feature_names must not be empty")
        metadata_columns = {} if metadata_columns is None else dict(metadata_columns)
        self._metadata_column_index = metadata_columns
        self._metadata_columns = frozenset(metadata_columns)
        if any(
            not isinstance(name, str) or not name for name in self._metadata_columns
        ):
            raise ValueError("metadata_columns must contain non-empty strings")
        with open(model_config_json, "rb") as f:
            model_config_bytes = f.read()
        model_config = orjson.loads(model_config_bytes)
        no_use_names = frozenset(model_config.get("no_use_feature", ()))
        if not all(isinstance(name, str) and name for name in no_use_names):
            raise ValueError("no_use_feature must contain non-empty feature names")
        selected_no_use = sorted(model_input_names & no_use_names)
        if selected_no_use:
            raise ValueError(
                "model_structure selects legacy no_use_feature values: "
                f"{selected_no_use}"
            )

        self._feature_column_index: dict[str, int] = model_config[
            "feature_column_index"
        ]

        sections: list[FeatureSection] = [
            AsItIsSection(),
            SharedEmbeddingSection(),
            HashBucketSection(),
            CutBoundarySection(),
            FixedWidthSection(),
            CategorySection(category_values or {}, frozenset(auto_category_features)),
        ]

        encoding_plan = self._build_encoding_config_plan(sections, model_config)
        feature_tsv_path = Path(feature_tsv)
        conf_common_dir = feature_tsv_path.parent
        base_definitions = self._load_feature_tsv(feature_tsv)
        fea_cross_tsv = str(conf_common_dir / "fea_cross.tsv")
        fid_cross_tsv = str(conf_common_dir / "fid_cross.tsv")
        cross_definitions = self._merge_cross_definitions(
            self._load_cross_tsv(fea_cross_tsv, "fea"),
            self._load_cross_tsv(fid_cross_tsv, "fid"),
        )
        self._validate_identity_codes(base_definitions, cross_definitions)
        codes_by_name = self._index_codes_by_name(
            tuple(
                definition
                for candidates in base_definitions.values()
                for definition in candidates
            )
            + tuple(cross_definitions.values())
        )
        required_codes = {
            self._resolve_unique_name(name, codes_by_name) for name in model_input_names
        }
        required_codes, resolved_definitions, cross_order = (
            self._resolve_required_with_encoder_dependencies(
                required_codes,
                base_definitions,
                cross_definitions,
                codes_by_name,
                encoding_plan,
            )
        )
        self._validate_selected_names(resolved_definitions)

        def resolve_name(name: str) -> int:
            return self._resolve_unique_name(name, codes_by_name)

        code_to_encoder: dict[int, EncoderExpr] = {}
        for code in sorted(required_codes):
            definition = resolved_definitions[code]
            if definition.cross_type == "fid":
                from .encoding.base import SchemaOnlyEncoderExpr

                code_to_encoder[code] = SchemaOnlyEncoderExpr(
                    FeatureOutputKind.SPARSE_I64_FIDS
                )
                continue
            assignment = encoding_plan.assignments.get(definition.name)
            if assignment is None:
                raise ValueError(
                    f"feature {definition.name!r} (code={code}) has no encoder config"
                )
            encoder = assignment.section.build_encoder(
                EncoderBuildContext(
                    code=code,
                    length=definition.length,
                    dtype=definition.dtype,
                    resolve_name=resolve_name,
                ),
                assignment.arguments,
            )
            code_to_encoder[code] = encoder

        self._code_to_encoder = code_to_encoder
        self._definitions = resolved_definitions
        self._cross_order = cross_order
        self._post_fid_category_codes = frozenset(
            dependency_code
            for definition in resolved_definitions.values()
            if definition.cross_type == "fid"
            for dependency_code in definition.dependency_codes
            if isinstance(code_to_encoder[dependency_code], CategoryEncoderExpr)
        )
        config_digest = self._config_digest(
            feature_tsv_path=feature_tsv_path,
            model_config_bytes=model_config_bytes,
            model_input_names=model_input_names,
            cross_paths=(Path(fea_cross_tsv), Path(fid_cross_tsv)),
            category_values=category_values or {},
            auto_category_features=frozenset(auto_category_features),
            required_names={
                definition.name for definition in resolved_definitions.values()
            },
            encoder_groups={
                name: assignment.section.group
                for name, assignment in encoding_plan.assignments.items()
            },
        )
        self._feature_catalog = FeatureCatalog(
            self._build_feature_descriptors(),
            config_digest=config_digest,
            columns=self.build_vortex_schema().names,
        )

    # ------------------------------------------------------------------
    # ExprPlan-facing accessors
    # ------------------------------------------------------------------

    @property
    def metadata_column_index(self) -> Mapping[str, int]:
        return self._metadata_column_index

    def build_encode_specs(self) -> list[tuple[str, int, int, EncoderExpr]]:
        """Tuples consumed by ExprPlan: ``(col_name, slot, length, encoder)``.

        Fid-cross outputs are produced directly by ``FidCrossExpr`` and are
        therefore skipped here — their column already exists post-cross.
        Meta-only columns are also skipped (they pass through unmodified).
        """
        return self._build_encode_specs(legacy_fid_stage=False)

    def build_legacy_fid_audit_encode_specs(
        self,
    ) -> list[tuple[str, int, int, EncoderExpr]]:
        return self._build_encode_specs(legacy_fid_stage=True)

    def build_post_fid_encode_specs(self) -> list[tuple[str, int, int, EncoderExpr]]:
        specs: list[tuple[str, int, int, EncoderExpr]] = []
        for code in sorted(self._post_fid_category_codes):
            definition = self._definitions[code]
            specs.append(
                (definition.name, code, definition.length, self._code_to_encoder[code])
            )
        return specs

    def _build_encode_specs(
        self,
        *,
        legacy_fid_stage: bool,
    ) -> list[tuple[str, int, int, EncoderExpr]]:
        specs: list[tuple[str, int, int, EncoderExpr]] = []
        for code, encoder in self._code_to_encoder.items():
            definition = self._definitions[code]
            feat_name = definition.name
            if feat_name in self._metadata_columns:
                continue
            if not getattr(encoder, "requires_base_encode", True):
                continue
            if isinstance(encoder, CategoryEncoderExpr) and (
                legacy_fid_stage or code in self._post_fid_category_codes
            ):
                encoder = AutoCategoryEncoderExpr()
            specs.append((feat_name, code, definition.length, encoder))
        return specs

    def build_fea_cross_ops(self) -> list[FeaCrossExpr]:
        ops: list[FeaCrossExpr] = []
        for code in self._cross_order:
            definition = self._definitions[code]
            if definition.cross_type != "fea":
                continue
            cls_name = definition.cross_class
            if cls_name is None:
                raise RuntimeError(f"cross feature {code} has no cross class")
            dependencies = [
                self._definitions[dependency_code]
                for dependency_code in definition.dependency_codes
            ]
            dep_names = [dependency.name for dependency in dependencies]
            factory = get_fea_factory(cls_name)
            if cls_name == "feaHitCount":
                dep_lens = [dependency.length for dependency in dependencies]
                in_dtype1 = string_to_pa_type(dependencies[0].dtype)
                in_dtype2 = string_to_pa_type(dependencies[1].dtype)
                out_dtype = string_to_pa_type(definition.dtype)
                ops.append(
                    factory(
                        name=definition.name,
                        dep_names=dep_names,
                        list_left=dep_lens[0] > 1,
                        list_right=dep_lens[1] > 1,
                        in_dtype1=in_dtype1,
                        in_dtype2=in_dtype2,
                        out_dtype=out_dtype,
                    )
                )
            else:
                out_dtype = string_to_pa_type(definition.dtype)
                ops.append(
                    factory(
                        name=definition.name,
                        dep_names=dep_names,
                        out_dtype=out_dtype,
                    )
                )
        return ops

    def build_fid_cross_ops(self) -> list[FidCrossExpr]:
        """Fid crosses operate on already-encoded ``fid_<dependency_code>`` columns.

        The output is ``fid_<cross_code>`` so identity is independent of its
        human-readable name.
        ``dep_widths`` carries the Array width of each dep so the pure-Expr
        vector path can enumerate the cartesian product at plan-build time.
        """
        ops: list[FidCrossExpr] = []
        for code in self._cross_order:
            definition = self._definitions[code]
            if definition.cross_type != "fid":
                continue
            cls_name = definition.cross_class
            if cls_name is None:
                raise RuntimeError(f"cross feature {code} has no cross class")
            dependencies = [
                self._definitions[dependency_code]
                for dependency_code in definition.dependency_codes
            ]
            dep_lens = [dependency.length for dependency in dependencies]
            if cls_name == "fidCross2":
                get_fid_factory(cls_name)
                bit_sizes = [24, 24]
            elif cls_name == "fidCross3":
                get_fid_factory(cls_name)
                bit_sizes = [16, 16, 16]
            elif cls_name == "fidCross4":
                get_fid_factory(cls_name)
                bit_sizes = [12, 12, 12, 12]
            elif cls_name.startswith("fidCrossN_"):
                try:
                    bit_sizes = [int(x) for x in cls_name.split("_")[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"invalid generic FidCross class: {cls_name}"
                    ) from exc
            else:
                raise ValueError(f"Unknown class name for FidCross: {cls_name}")
            if len(bit_sizes) != len(dependencies):
                raise ValueError(
                    f"FidCross class {cls_name} has {len(bit_sizes)} bit sizes for "
                    f"{len(dependencies)} dependencies"
                )
            ops.append(
                FidCrossNExpr(
                    name=storage_column_name(code, FeatureRepresentation.FID),
                    dep_names=[
                        self._require_fid_dependency(dependency.code)
                        for dependency in dependencies
                    ],
                    bit_sizes=bit_sizes,
                    dep_widths=dep_lens,
                    slot=code,
                )
            )
        return ops

    def build_output_cols(self) -> list[str]:
        """Final column order — exactly matches ``build_vortex_schema().names``."""
        return [f.name for f in self.build_vortex_schema()]

    # ------------------------------------------------------------------
    # Vortex schema
    # ------------------------------------------------------------------

    def build_vortex_schema(self) -> pa.Schema:
        m = {b"vortex-encoding:compression": b"zstd"}
        # Explicit metadata columns pass through unchanged from the parser.
        fields: list[pa.Field] = [
            pa.field(name, pa.int32(), metadata=m)
            for name in sorted(self._metadata_columns)
        ]
        output_sources = {field.name: field.name for field in fields}

        for code in sorted(self._definitions):
            definition = self._definitions[code]
            feat_name = definition.name
            if feat_name in self._metadata_columns:
                continue

            encoder = self.get_encoder(code)

            length = definition.length

            field_type = output_kind_pa_type(encoder.output_kind, length)

            if encoder.output_kind is FeatureOutputKind.DENSE_FLOAT:
                col_name = storage_column_name(code, FeatureRepresentation.DENSE)
            else:
                col_name = storage_column_name(code, FeatureRepresentation.FID)

            existing_source = output_sources.get(col_name)
            if existing_source is not None:
                raise ValueError(
                    "physical feature column collision: "
                    f"{existing_source!r} and {feat_name!r} -> {col_name!r}"
                )
            output_sources[col_name] = feat_name

            field_meta = m.copy()
            if length > 1:
                field_meta[b"pad_length"] = str(length).encode("utf-8")
                pad_value = encoder.encoded_default(code)
                field_meta[b"pad_value"] = str(pad_value).encode("utf-8")

            fields.append(pa.field(col_name, field_type, metadata=field_meta))

        return pa.schema(fields)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def feature_column_index(self) -> dict[str, int]:
        return self._feature_column_index

    @property
    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return tuple(self._definitions[code] for code in sorted(self._definitions))

    @property
    def feature_catalog(self) -> FeatureCatalog:
        return self._feature_catalog

    def get_encoder(self, code: int) -> EncoderExpr:
        try:
            return self._code_to_encoder[code]
        except KeyError as exc:
            raise ValueError(f"feature code {code} has no encoder") from exc

    # ------------------------------------------------------------------
    # Internal loaders / resolvers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_encoding_config_plan(
        sections: Iterable[FeatureSection],
        model_config: dict,
    ) -> EncodingConfigPlan:
        assignments: dict[str, EncodingAssignment] = {}
        for section in sections:
            section_assignments = section.collect(model_config)
            for name, arguments in section_assignments.items():
                candidate = EncodingAssignment(section, arguments)
                existing = assignments.get(name)
                if existing is None:
                    assignments[name] = candidate
                    continue
                existing_priority = existing.section.priority_for(existing.arguments)
                candidate_priority = section.priority_for(arguments)
                if candidate_priority > existing_priority:
                    assignments[name] = candidate
                elif candidate_priority == existing_priority:
                    raise ValueError(
                        f"feature {name!r} has multiple encoder configs: "
                        f"{existing.section.group.value}, {section.group.value}"
                    )
        return EncodingConfigPlan(assignments=assignments)

    @staticmethod
    def _load_feature_tsv(path: str) -> dict[int, tuple[FeatureDefinition, ...]]:
        candidates_by_code: dict[int, set[FeatureDefinition]] = defaultdict(set)
        with open(path, encoding="utf-8") as source:
            header = source.readline().rstrip("\n").split("\t")
            if header[:4] != ["code", "feature", "length", "type"]:
                raise ValueError(f"invalid feature TSV header in {path}")
            for line_number, line in enumerate(source, start=2):
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    raise ValueError(f"invalid feature TSV row {path}:{line_number}")
                code_text, name, length_text, dtype = (
                    part.strip() for part in parts[:4]
                )
                try:
                    definition = FeatureDefinition(
                        code=int(code_text),
                        name=name,
                        length=int(length_text),
                        dtype=dtype,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"invalid feature TSV row {path}:{line_number}"
                    ) from exc
                if not definition.name or not definition.dtype:
                    raise ValueError(f"invalid feature TSV row {path}:{line_number}")
                candidates_by_code[definition.code].add(definition)
        return {
            code: tuple(sorted(candidates, key=repr))
            for code, candidates in candidates_by_code.items()
        }

    @staticmethod
    def _merge_cross_definitions(
        *sources: Mapping[int, CrossDefinition],
    ) -> dict[int, CrossDefinition]:
        result: dict[int, CrossDefinition] = {}
        for source in sources:
            for code, definition in source.items():
                existing = result.get(code)
                if existing is not None and existing != definition:
                    raise ValueError(
                        f"conflicting cross feature code {code}: "
                        f"{existing!r}, {definition!r}"
                    )
                result[code] = definition
        return result

    @staticmethod
    def _validate_identity_codes(
        base_definitions: Mapping[int, tuple[FeatureDefinition, ...]],
        cross_definitions: Mapping[int, CrossDefinition],
    ) -> None:
        collisions = sorted(set(base_definitions) & set(cross_definitions))
        if collisions:
            raise ValueError(
                f"feature codes declared as both base and cross: {collisions}"
            )

    @staticmethod
    def _index_codes_by_name(
        definitions: Sequence[FeatureDefinition | CrossDefinition],
    ) -> dict[str, tuple[int, ...]]:
        codes: dict[str, set[int]] = defaultdict(set)
        for definition in definitions:
            codes[definition.name].add(definition.code)
        return {
            name: tuple(sorted(feature_codes)) for name, feature_codes in codes.items()
        }

    @staticmethod
    def _resolve_unique_name(
        name: str,
        codes_by_name: Mapping[str, tuple[int, ...]],
    ) -> int:
        codes = codes_by_name.get(name, ())
        if not codes:
            raise ValueError(f"feature name {name!r} is not declared in feature.tsv")
        if len(codes) != 1:
            raise ValueError(f"ambiguous feature name {name!r}: codes={list(codes)}")
        return codes[0]

    @classmethod
    def _resolve_required_with_encoder_dependencies(
        cls,
        requested_codes: set[int],
        base_definitions: Mapping[int, tuple[FeatureDefinition, ...]],
        cross_definitions: Mapping[int, CrossDefinition],
        codes_by_name: Mapping[str, tuple[int, ...]],
        encoding_plan: EncodingConfigPlan,
    ) -> tuple[set[int], dict[int, FeatureDefinition], list[int]]:
        required_codes = set(requested_codes)
        while True:
            resolved, cross_order = cls._resolve_required_definitions(
                required_codes,
                base_definitions,
                cross_definitions,
            )
            required_codes.update(resolved)
            encoder_dependencies: set[int] = set()
            for definition in resolved.values():
                assignment = encoding_plan.assignments.get(definition.name)
                if (
                    assignment is None
                    or assignment.section.group is not FeatureGroup.SHARED_EMBEDDING
                ):
                    continue
                target_name = assignment.arguments.get("target_name")
                if not isinstance(target_name, str) or not target_name:
                    raise ValueError(
                        f"shared embedding feature {definition.name!r} "
                        "requires target_name"
                    )
                encoder_dependencies.add(
                    cls._resolve_unique_name(target_name, codes_by_name)
                )
            new_codes = encoder_dependencies - required_codes
            if not new_codes:
                return required_codes, resolved, cross_order
            required_codes.update(new_codes)

    @classmethod
    def _resolve_required_definitions(
        cls,
        requested_codes: set[int],
        base_definitions: Mapping[int, tuple[FeatureDefinition, ...]],
        cross_definitions: Mapping[int, CrossDefinition],
    ) -> tuple[dict[int, FeatureDefinition], list[int]]:
        resolved: dict[int, FeatureDefinition] = {}
        resolving: set[int] = set()
        cross_order: list[int] = []

        def resolve(code: int) -> FeatureDefinition:
            cached = resolved.get(code)
            if cached is not None:
                return cached
            base_candidates = base_definitions.get(code)
            if base_candidates is not None:
                if len(base_candidates) != 1:
                    raise ValueError(
                        f"selected feature code {code} has conflicting definitions: "
                        f"{list(base_candidates)!r}"
                    )
                base = base_candidates[0]
                cls._validate_selected_base_definition(base)
                resolved[code] = base
                return base
            cross = cross_definitions.get(code)
            if cross is None:
                raise ValueError(f"feature code {code} is not declared")
            if code in resolving:
                raise ValueError(f"cyclic cross dependency at feature code {code}")
            resolving.add(code)
            dependencies = tuple(resolve(item) for item in cross.dependency_codes)
            length = cls._cross_output_length(cross, dependencies)
            definition = FeatureDefinition(
                code=cross.code,
                name=cross.name,
                length=length,
                dtype="float32" if cross.cross_type == "fea" else "int64",
                cross_type=cross.cross_type,
                cross_class=cross.cross_class,
                dependency_codes=cross.dependency_codes,
            )
            resolving.remove(code)
            resolved[code] = definition
            cross_order.append(code)
            return definition

        for code in sorted(requested_codes):
            resolve(code)
        return resolved, cross_order

    @staticmethod
    def _validate_selected_base_definition(definition: FeatureDefinition) -> None:
        if definition.length <= 0:
            raise ValueError(
                f"selected feature {definition.name!r} (code={definition.code}) "
                f"has invalid length {definition.length}"
            )
        dtype = string_to_pa_type(definition.dtype)
        if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
            raise ValueError(
                f"selected feature {definition.name!r} (code={definition.code}) "
                f"has unsupported parser dtype {definition.dtype!r}"
            )

    @staticmethod
    def _cross_output_length(
        cross: CrossDefinition,
        dependencies: Sequence[FeatureDefinition],
    ) -> int:
        if not dependencies:
            raise ValueError(f"cross feature {cross.code} has no dependencies")
        if cross.cross_type == "fea":
            if len(dependencies) != 2:
                raise ValueError(
                    f"fea cross {cross.code} requires two dependencies, "
                    f"got {len(dependencies)}"
                )
            if cross.cross_class in {"feaCrossSub", "feaCrossDiv"}:
                left, right = dependencies
                if left.length != right.length:
                    raise ValueError(
                        f"feature {cross.name!r} has mismatched dependency lengths: "
                        f"{left.length} vs {right.length}"
                    )
                return left.length
            if cross.cross_class == "feaHitCount":
                return 1
            raise ValueError(f"unknown fea cross class: {cross.cross_class}")
        length = 1
        for dependency in dependencies:
            length *= dependency.length
        return length

    @staticmethod
    def _validate_selected_names(
        definitions: Mapping[int, FeatureDefinition],
    ) -> None:
        codes_by_materialized_name: dict[str, list[int]] = defaultdict(list)
        for definition in definitions.values():
            if definition.cross_type != "fid":
                codes_by_materialized_name[definition.name].append(definition.code)
        collisions = {
            name: sorted(codes)
            for name, codes in codes_by_materialized_name.items()
            if len(codes) > 1
        }
        if collisions:
            raise ValueError(
                f"selected features have ambiguous materialization names: {collisions}"
            )

    def _require_fid_dependency(self, code: int) -> str:
        encoder = self.get_encoder(code)
        if encoder.output_kind is FeatureOutputKind.DENSE_FLOAT:
            raise ValueError(f"fid cross dependency {code} has dense representation")
        return storage_column_name(code, FeatureRepresentation.FID)

    def _build_feature_descriptors(self) -> tuple[FeatureDescriptor, ...]:
        descriptors: list[FeatureDescriptor] = []
        for code in sorted(self._definitions):
            definition = self._definitions[code]
            encoder = self.get_encoder(code)
            representation = (
                FeatureRepresentation.DENSE
                if encoder.output_kind is FeatureOutputKind.DENSE_FLOAT
                else FeatureRepresentation.FID
            )
            descriptors.append(
                FeatureDescriptor(
                    code=code,
                    name=definition.name,
                    representation=representation,
                    length=definition.length,
                    output_kind=encoder.output_kind,
                    encoder=type(encoder).__name__,
                    cross_type=definition.cross_type,
                    cross_class=definition.cross_class,
                    dependency_codes=definition.dependency_codes,
                )
            )
        return tuple(descriptors)

    @staticmethod
    def _config_digest(
        *,
        feature_tsv_path: Path,
        model_config_bytes: bytes,
        model_input_names: frozenset[str],
        cross_paths: Sequence[Path],
        category_values: Mapping[str, Mapping[int, int]],
        auto_category_features: frozenset[str],
        required_names: set[str],
        encoder_groups: Mapping[str, FeatureGroup],
    ) -> str:
        digest = hashlib.sha256()

        def update(label: str, payload: bytes) -> None:
            label_bytes = label.encode("utf-8")
            digest.update(len(label_bytes).to_bytes(8, "big"))
            digest.update(label_bytes)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)

        update("feature.tsv", feature_tsv_path.read_bytes())
        update("model_config.json", model_config_bytes)
        update(
            "model_input_feature_names",
            orjson.dumps(sorted(model_input_names)),
        )
        for path in cross_paths:
            update(path.name, path.read_bytes())
        cut_boundaries = orjson.loads(model_config_bytes).get(
            "feature_cut_boundary", {}
        )
        for name in sorted(required_names):
            if encoder_groups.get(name) is not FeatureGroup.CATEGORY:
                continue
            if name in auto_category_features:
                continue
            if name in cut_boundaries:
                continue
            mapping = category_values.get(name)
            if mapping is None:
                raise ValueError(
                    f"category feature {name!r} has no Recipe-owned category mapping"
                )
            update(f"category:{name}", orjson.dumps(sorted(mapping.items())))
        return digest.hexdigest()

    @staticmethod
    def _load_cross_tsv(path: str, cross_type: str) -> dict[int, CrossDefinition]:
        if cross_type not in {"fea", "fid"}:
            raise ValueError(f"unsupported cross type: {cross_type!r}")
        result: dict[int, CrossDefinition] = {}
        with open(path, "r", encoding="utf-8") as f:
            if not f.readline():
                raise ValueError(f"cross TSV is empty: {path}")
            for line_number, line in enumerate(f, start=2):
                if not line.strip():
                    continue
                parts = [p.strip() for p in re.split(r"[,\t]+", line)]
                if len(parts) < 3:
                    raise ValueError(f"invalid cross TSV row {path}:{line_number}")
                try:
                    code = int(parts[0])
                    dependency_codes = tuple(
                        int(float(value)) for value in parts[3:] if value
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"invalid cross TSV row {path}:{line_number}"
                    ) from exc
                definition = CrossDefinition(
                    code=code,
                    name=parts[1],
                    cross_type=cross_type,
                    cross_class=parts[2],
                    dependency_codes=dependency_codes,
                )
                if not definition.name or not definition.cross_class:
                    raise ValueError(f"invalid cross TSV row {path}:{line_number}")
                existing = result.get(code)
                if existing is not None and existing != definition:
                    raise ValueError(
                        f"conflicting cross feature code {code} in {path}: "
                        f"{existing!r}, {definition!r}"
                    )
                result[code] = definition
        return result
