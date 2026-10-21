from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FeaturePartition:
    feature_codes: dict[str, int]
    feature_lengths: dict[str, int]
    categorical_cardinalities: dict[str, int]
    sparse_features: tuple[str, ...]


def compile_feature_partition(
    feature_names: Iterable[str],
    *,
    conf_root: Path,
    model_conf_dir: Path,
    category_values: Mapping[str, Mapping[int, int]] | None = None,
    auto_category_features: Iterable[str] = (),
) -> FeaturePartition:
    from flare.data.pipeline.production.builder.row_parser import (
        build_feature_processor,
    )
    from flare.data.schema.feature_catalog import FeatureOutputKind

    requested = tuple(dict.fromkeys(feature_names))
    processor = build_feature_processor(
        conf_root,
        model_conf_dir,
        category_values=category_values,
        auto_category_features=auto_category_features,
    )
    feature_codes: dict[str, int] = {}
    feature_lengths: dict[str, int] = {}
    categorical_cardinalities: dict[str, int] = {}
    sparse_features: list[str] = []
    for name in requested:
        code = processor.feature_catalog.resolve_unique_name(name)
        descriptor = processor.feature_catalog.require(code)
        encoder = processor.get_encoder(code)
        feature_codes[name] = code
        feature_lengths[name] = descriptor.length
        if encoder.output_kind is FeatureOutputKind.DENSE_I32_INDICES:
            cardinality = encoder.encoded_cardinality
            if cardinality is None or cardinality <= 0:
                raise ValueError(
                    f"Categorical feature {name!r} has invalid encoded cardinality: "
                    f"{cardinality!r}"
                )
            categorical_cardinalities[name] = cardinality
        elif encoder.output_kind is FeatureOutputKind.SPARSE_I64_FIDS:
            sparse_features.append(name)
        elif encoder.output_kind is FeatureOutputKind.DENSE_FLOAT:
            raise ValueError(
                f"Feature partition does not support floating-point feature {name!r}"
            )
        else:
            raise ValueError(
                f"Feature {name!r} has unknown output kind {encoder.output_kind!r}"
            )
    return FeaturePartition(
        feature_codes=feature_codes,
        feature_lengths=feature_lengths,
        categorical_cardinalities=categorical_cardinalities,
        sparse_features=tuple(sparse_features),
    )


__all__ = ["FeaturePartition", "compile_feature_partition"]
