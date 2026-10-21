import importlib
import os
from pathlib import Path
from types import ModuleType


def find_named_directory(root: Path, target_name: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Search root directory not found: {root}")
    if not target_name:
        raise ValueError("target_name must be a non-empty string")
    matches = [p for p in root.rglob(target_name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Cannot find directory {target_name!r} under {root}")
    if len(matches) > 1:
        relative = [str(p.relative_to(root)) for p in matches]
        raise ValueError(
            f"Found ambiguous matches for {target_name!r} under {root}: {relative}"
        )
    return matches[0]


def load_model_structure(
    conf_root: Path,
    conf_version: str,
    base_module: str,
) -> ModuleType:
    if not isinstance(conf_root, Path):
        conf_root = Path(conf_root)
    if not base_module:
        raise ValueError("base_module must be a non-empty string")
    conf_dir = find_named_directory(conf_root, conf_version)
    rel_path = str(conf_dir.relative_to(conf_root)).replace(os.sep, ".")
    module_path = f"{base_module}.{rel_path}.model_structure"
    return importlib.import_module(module_path)


__all__ = ["find_named_directory", "load_model_structure"]
