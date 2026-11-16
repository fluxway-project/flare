"""Explicit native sparse storage limits shared by training and evaluation."""

import tempfile
from pathlib import Path
from typing import Any


def sparse_storage_options(spec: dict[str, Any], *, group: str) -> dict[str, Any]:
    import sparse_embedding as se

    limit = spec.get("sparse_storage_memory_limit_bytes")
    spill_limit = spec.get("sparse_storage_max_spill_bytes")
    spill_root = spec.get("sparse_storage_spill_dir")
    max_pages = spec.get("sparse_storage_max_pages", 0)
    entries_per_page = spec.get("sparse_storage_entries_per_page", 0)
    for name, value in (("memory limit", limit), ("spill limit", spill_limit)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"sparse storage {name} must be a positive integer")
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 0:
        raise ValueError("sparse storage max_pages must be a non-negative integer")
    if entries_per_page not in (0, 4096, 8192, 16384):
        raise ValueError(
            "sparse storage entries_per_page must be 0, 4096, 8192 or 16384"
        )
    if spill_limit is not None and spill_root is None:
        raise ValueError("sparse storage spill limit requires a spill directory")
    directory = None
    if spill_root is not None:
        root = Path(spill_root)
        root.mkdir(parents=True, exist_ok=True)
        # Separate groups, ranks, and restored actors must never share spill files.
        directory = tempfile.mkdtemp(prefix="worker-", dir=root)
    result = {
        "storage": se.AutoStorage(
            memory_limit_bytes=limit,
            spill_directory=directory,
            max_spill_bytes=spill_limit,
        )
    }
    if max_pages or entries_per_page:
        result["storage_tuning"] = se.StorageTuningOptions(
            max_pages=max_pages,
            entries_per_page=entries_per_page,
            calibrate_page_geometry=not bool(entries_per_page),
        )
    if limit is not None or directory is not None or max_pages:
        print(
            f"[SparseStorage] group={group} memory_limit_bytes={limit} "
            f"max_pages={max_pages} spill_directory={directory} "
            f"max_spill_bytes={spill_limit}",
            flush=True,
        )
    return result
