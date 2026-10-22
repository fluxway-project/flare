from __future__ import annotations

import ast
from pathlib import Path

_SOURCE_ROOT = Path(__file__).parents[2] / "src" / "flare"


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


def test_public_layer_does_not_export_ranking_benchmark_helpers() -> None:
    public_benchmark_root = _SOURCE_ROOT / "runtime" / "fluxway" / "benchmark"
    public_throughput_module = (
        _SOURCE_ROOT / "runtime" / "fluxway" / "placement" / "throughput.py"
    )
    assert not public_benchmark_root.exists()
    assert not public_throughput_module.exists()


def test_shared_recipe_has_no_imports_outside_its_package() -> None:
    root = _SOURCE_ROOT.parents[1] / "recipes/recsys"
    violations = []
    for path in sorted(root.rglob("*")):
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source, filename=str(path))):
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if module.startswith("recipes.") and not module.startswith(
                    "recipes.recsys."
                ):
                    violations.append(f"{path}:{node.lineno}: {module}")
    assert not violations
