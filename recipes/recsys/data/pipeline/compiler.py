from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from flare.data.pipeline.production.processing.artifact import (
    FeatureProcessingArtifact,
    ProcessingArtifactCompiler,
)
from flare.data.pipeline.production.processing.parser_contract import ParserContract

CategoryLoader = Callable[
    [Path, Path], tuple[Mapping[str, Mapping[int | float, int]], frozenset[str]]
]


def compile_processing_artifact(
    *,
    conf_dir: Path,
    model_conf_dir: Path,
    category_dir: Path,
    parser_contract: ParserContract,
    category_loader: CategoryLoader,
) -> FeatureProcessingArtifact:
    category_values, auto_category_features = category_loader(
        model_conf_dir,
        category_dir,
    )
    return ProcessingArtifactCompiler().compile(
        conf_dir=conf_dir,
        model_conf_dir=model_conf_dir,
        parser_contract=parser_contract,
        category_values=category_values,
        auto_category_features=auto_category_features,
    )


__all__ = ["CategoryLoader", "compile_processing_artifact"]
