from __future__ import annotations

import ast
from pathlib import Path

_SOURCE_ROOT = Path(__file__).parents[2] / "src" / "flare"
_PUBLIC_DOCS_ROOT = Path(__file__).parents[2] / "docs"
_PRIVATE_MARKERS = (
    "ae11",
    "whw",
    "weizhipin",
    "boss",
    "geek",
    "recsys",
    "ranking",
    "auc",
    "gauc",
    "deal_type",
    "detailed_deal_type",
)


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted(_SOURCE_ROOT.rglob("*.py")))


def test_public_layer_does_not_import_recipe_packages() -> None:
    violations: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                module = next((alias.name for alias in node.names if alias.name), None)
            elif isinstance(node, ast.ImportFrom):
                module = node.module
            if module is not None and (
                module == "recipes" or module.startswith("recipes.")
            ):
                violations.append(f"{path}: {module}")
    assert violations == []


def test_public_layer_has_no_recipe_specific_markers() -> None:
    violations = []
    for path in _python_files():
        lowered = path.read_text(encoding="utf-8").lower()
        for marker in _PRIVATE_MARKERS:
            if marker in lowered:
                violations.append(f"{path}: {marker}")
    assert violations == []


def test_public_docs_have_no_recipe_specific_markers() -> None:
    violations = []
    for path in sorted(_PUBLIC_DOCS_ROOT.glob("*.md")):
        lowered = path.read_text(encoding="utf-8").lower()
        for marker in _PRIVATE_MARKERS:
            if marker in lowered:
                violations.append(f"{path}: {marker}")
    assert violations == []


def test_public_layer_does_not_export_ranking_benchmark_helpers() -> None:
    public_benchmark_root = _SOURCE_ROOT / "runtime" / "fluxway" / "benchmark"
    public_throughput_module = (
        _SOURCE_ROOT / "runtime" / "fluxway" / "placement" / "throughput.py"
    )
    assert not public_benchmark_root.exists()
    assert not public_throughput_module.exists()
