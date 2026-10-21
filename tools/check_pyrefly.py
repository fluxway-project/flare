from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import tyro

_SOURCE_PACKAGES = ("fluxway", "jax", "flax")


def package_search_path(package_name: str) -> Path:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(f"Pyrefly requires an importable {package_name} package")
    locations = tuple(
        Path(location).resolve() for location in spec.submodule_search_locations
    )
    if len(locations) != 1:
        raise RuntimeError(
            f"Pyrefly requires exactly one {package_name} package location, got {locations!r}"
        )
    package_root = locations[0]
    if not (package_root / "__init__.py").is_file():
        raise RuntimeError(
            f"Pyrefly package root does not contain __init__.py: {package_root}"
        )
    return package_root.parent


def source_search_paths() -> tuple[Path, ...]:
    return tuple(package_search_path(package_name) for package_name in _SOURCE_PACKAGES)


@dataclass(frozen=True)
class CheckPyreflyArgs:
    paths: list[str] = field(default_factory=lambda: ["."])


def main() -> None:
    arguments = tyro.cli(CheckPyreflyArgs)
    executable = shutil.which("pyrefly")
    if executable is None:
        raise RuntimeError("Pyrefly executable is required")
    command = [executable, "check"]
    for search_path in source_search_paths():
        command.extend(("--search-path", str(search_path)))
    command.extend(arguments.paths)
    completed = subprocess.run(command, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
