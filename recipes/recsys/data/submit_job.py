"""Package tracked recipe code with native extensions and submit a direct Ray Job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
NATIVE_MODULES = (
    Path("src/flare/data/pipeline/production/builder/_fast_parse"),
    Path("src/flare/data/pipeline/production/processing/native/_scan"),
)


def _bundle_paths(repository: Path, recipe: str) -> tuple[Path, ...]:
    selected = (
        "recipes/__init__.py",
        "recipes/config",
        "src/flare",
        "recipes/recsys",
        f"recipes/{recipe}",
    )
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *selected],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    paths = {Path(name.decode()) for name in result.stdout.split(b"\0") if name}
    for module in NATIVE_MODULES:
        candidates = [
            module.parent / (module.name + suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            if (repository / (module.parent / (module.name + suffix))).is_file()
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Build exactly one native extension before submitting: {module}"
            )
        paths.add(candidates[0])
    selected_paths = []
    for path in sorted(paths):
        source = repository / path
        if source.is_symlink():
            raise ValueError(f"Ray bundle must not follow a symbolic link: {path}")
        if source.is_file():
            selected_paths.append(path)
    return tuple(selected_paths)


def build_bundle(repository: Path, recipe: str, bundle_dir: Path) -> Path:
    """Package only Git-tracked sources plus the two required native modules."""
    paths = _bundle_paths(repository, recipe)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(hashlib.sha256((repository / path).read_bytes()).digest())
    bundle_dir.mkdir(parents=True, exist_ok=True)
    destination = bundle_dir / f"flare-{recipe}-{digest.hexdigest()[:16]}.zip"
    if destination.exists():
        return destination
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".zip", dir=bundle_dir
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as bundle:
            for path in paths:
                bundle.write(repository / path, arcname=path.as_posix())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True, help="Ray Jobs API address")
    parser.add_argument(
        "--recipe", required=True, help="Application package under recipes/"
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--submission-id")
    parser.add_argument("--wait", action="store_true")
    if "--" not in arguments:
        if "-h" in arguments or "--help" in arguments:
            parser.parse_args(arguments)
        parser.error("separate recipe options with --")
    split = arguments.index("--")
    options = parser.parse_args(arguments[:split])
    if not options.recipe.isidentifier():
        parser.error("--recipe must be a Python package name")
    recipe_options = arguments[split + 1 :]
    if not recipe_options:
        parser.error("recipe options are required after --")

    bundle = build_bundle(REPOSITORY, options.recipe, options.bundle_dir.resolve())
    runtime = {
        "env_vars": {
            "PYTHONPATH": "src:.",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    }
    command = [
        "ray",
        "job",
        "submit",
        "--address",
        options.address,
        "--entrypoint-num-cpus",
        "1",
        "--working-dir",
        str(bundle),
        "--runtime-env-json",
        json.dumps(runtime, separators=(",", ":")),
    ]
    if options.submission_id is not None:
        command.extend(("--submission-id", options.submission_id))
    if not options.wait:
        command.append("--no-wait")
    command.extend(("--", "python", "-m", f"recipes.{options.recipe}.data.run"))
    command.extend(recipe_options)
    print(f"Submitting {options.recipe} from {bundle}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
