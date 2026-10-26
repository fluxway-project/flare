from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

_HAS_RECIPES = importlib.util.find_spec("recipes") is not None


def _imports_recipes(file_path: Path) -> bool:
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "recipes" or alias.name.startswith("recipes."):
                    return True
        elif isinstance(node, ast.ImportFrom) and (
            node.module == "recipes"
            or (node.module and node.module.startswith("recipes."))
        ):
            return True
    return False


def pytest_ignore_collect(collection_path: Any, config: Any) -> bool:
    if _HAS_RECIPES:
        return False
    path = Path(str(collection_path))
    return bool(path.suffix == ".py" and _imports_recipes(path))
