"""Audit source boundaries and probe the installed owner in temporary storage."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _run(command: list[str], cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def _check_roots(roots: dict[str, Path]) -> list[CheckResult]:
    return [
        CheckResult(
            name=f"{name} root",
            passed=path.is_dir() and (path / ".git").exists(),
            detail=str(path),
        )
        for name, path in roots.items()
    ]


def _check_legacy_sparse_wire(sparse_root: Path) -> CheckResult:
    path = sparse_root / "sparse_embedding/checkpoint/restore_options.py"
    if not path.is_file():
        return CheckResult("sparse owner wire source", False, str(path))
    source = path.read_text(encoding="utf-8")
    markers = (
        '"router_id"',
        '"key_encoding"',
        '"dimension_scope"',
        '"static_config"',
        '"target_partition_id"',
    )
    present = tuple(
        marker.strip('"') for marker in markers if re.search(re.escape(marker), source)
    )
    return CheckResult(
        "sparse owner target wire",
        not present,
        "legacy fields present: " + ", ".join(present)
        if present
        else "target field names not found",
    )


def _check_fluxway_duplicate_routing(fluxway_root: Path) -> CheckResult:
    module = fluxway_root / "fluxway/integrations/sparse_embedding/routing.py"
    symbols = (
        "DynamicRoutingRegistry",
        "RouterKernel",
        "ModuloRouterKernel",
        "NormalizedRouterDescriptor",
    )
    if not module.is_file():
        return CheckResult("Fluxway sparse routing duplicate", True, "module absent")
    source = module.read_text(encoding="utf-8")
    present = tuple(symbol for symbol in symbols if symbol in source)
    return CheckResult(
        "Fluxway sparse routing duplicate",
        not present,
        "duplicate symbols present: " + ", ".join(present)
        if present
        else "module contains no duplicate symbols",
    )


def _check_installed_sparse_owner(
    python: Path, cwd: Path, sparse_root: Path
) -> CheckResult:
    probe = """
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import sparse_embedding as se
from sparse_embedding.checkpoint import (
    SUPPORTED_KEY_DTYPES, RestorePlan, PartitionSpec,
    LocalLanceStorage, SparseCheckpoint,
)
def verify_native_restore(key_dtype, root):
    options = RestorePlan.from_partition(
        spec=PartitionSpec.modulo(),
        partition_count=1, partition_id=0, key_dtype=key_dtype,
    )
    config = se.EmbeddingConfig(
        dims=[2], thread_budget=1, key_dtype=key_dtype,
        value_dtype="float32", enable_optimizer=True,
    )
    source = se.EmbeddingStore(config, optimizer_config=se.Adam(lr=0.01))
    if source.key_dtype() != key_dtype:
        raise RuntimeError("native store key dtype does not match public API")
    keys = np.asarray([0, 1], dtype=key_dtype)
    values = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    source.assign(keys, values)
    storage = LocalLanceStorage(root)
    try:
        checkpoint = SparseCheckpoint(storage)
        saved = checkpoint.save_training_checkpoint(source, "installed-probe")
        target = se.EmbeddingStore(config, optimizer_config=se.Adam(lr=0.01))
        checkpoint.load_training_checkpoint(target, saved.handle, options)
        restored, _ = target.lookup(keys, dim=2)
        np.testing.assert_array_equal(np.from_dlpack(restored).reshape(values.shape), values)
    finally:
        storage.close()
with TemporaryDirectory(prefix="ae11-installed-owner-") as directory:
    for key_dtype in sorted(SUPPORTED_KEY_DTYPES):
        verify_native_restore(key_dtype, Path(directory) / key_dtype)
print(json.dumps({"module": str(Path(se.__file__).resolve()),
                  "key_dtypes": sorted(SUPPORTED_KEY_DTYPES)}))
"""
    code, output = _run([str(python), "-c", probe], cwd)
    if code != 0:
        return CheckResult("installed sparse owner API/native", False, output)
    evidence = json.loads(output)
    module_path = Path(evidence["module"]).resolve()
    sparse_root_resolved = sparse_root.resolve()
    if (
        sparse_root_resolved in module_path.parents
        or module_path == sparse_root_resolved
    ):
        return CheckResult(
            "installed sparse owner API/native",
            False,
            f"loaded from source checkout ({module_path}), not an isolated wheel installation in site-packages",
        )
    if "site-packages" not in str(module_path):
        return CheckResult(
            "installed sparse owner API/native",
            False,
            f"module path ({module_path}) is not located in site-packages",
        )
    return CheckResult("installed sparse owner API/native", True, str(evidence))


def _check_flare_private_boundaries(flare_root: Path) -> CheckResult:
    roots = (
        flare_root / "src/flare/inference",
        flare_root / "src/flare/runtime/fluxway",
        flare_root / "recipes/ae11",
        flare_root / "recipes/whw",
    )
    forbidden = (
        "sparse_embedding._core.checkpoint",
        "import_batch_dlpack",
        "export_batches_dlpack",
        "SparseWeightSnapshot",
        "restore_weight_snapshot",
    )
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for marker in forbidden:
                if marker in source:
                    violations.append(f"{path}:{marker}")
    return CheckResult(
        "Flare private owner boundaries",
        not violations,
        "; ".join(violations) if violations else "no forbidden symbols",
    )


def _check_flare_router_owner(flare_root: Path) -> CheckResult:
    path = flare_root / "src/flare/inference/weights/router.py"
    if not path.is_file():
        return CheckResult("Flare duplicate router owner", True, "router module absent")
    source = path.read_text(encoding="utf-8")
    markers = tuple(
        marker
        for marker in (
            "class RouterDescriptor",
            "class RouterBackendRegistry",
            "def route_keys",
        )
        if marker in source
    )
    return CheckResult(
        "Flare duplicate router owner",
        not markers,
        "legacy router symbols present: " + ", ".join(markers)
        if markers
        else "no legacy router symbols",
    )


def _check_git_status(repo: Path, name: str) -> CheckResult:
    code, output = _run(["git", "status", "--short"], repo)
    entries = output.splitlines() if output else []
    sample = ", ".join(entries[:5])
    return CheckResult(
        f"{name} git status",
        code == 0,
        "clean"
        if code == 0 and not entries
        else f"{len(entries)} changed entries; sample: {sample}",
    )


def main() -> int:
    flare_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flare", type=Path, default=flare_root)
    parser.add_argument(
        "--sparse", type=Path, default=flare_root.parent / "sparse_embedding"
    )
    parser.add_argument("--fluxway", type=Path, default=flare_root.parent / "fluxway")
    parser.add_argument("--owner-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    roots = {
        "flare-jax": args.flare.resolve(),
        "sparse_embedding": args.sparse.resolve(),
        "fluxway": args.fluxway.resolve(),
    }
    results = _check_roots(roots)
    if all(result.passed for result in results):
        results.extend(
            (
                _check_git_status(roots["sparse_embedding"], "sparse_embedding"),
                _check_git_status(roots["fluxway"], "fluxway"),
                _check_legacy_sparse_wire(roots["sparse_embedding"]),
                _check_fluxway_duplicate_routing(roots["fluxway"]),
                _check_installed_sparse_owner(
                    args.owner_python.expanduser().absolute(),
                    roots["flare-jax"],
                    roots["sparse_embedding"],
                ),
                _check_flare_private_boundaries(roots["flare-jax"]),
                _check_flare_router_owner(roots["flare-jax"]),
            )
        )
    for result in results:
        status = "PASS" if result.passed else "BLOCKED"
        print(f"[{status}] {result.name}: {result.detail}")
    return 0 if all(result.passed for result in results) else 2


if __name__ == "__main__":
    sys.exit(main())
