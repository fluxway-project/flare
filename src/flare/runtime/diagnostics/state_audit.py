from __future__ import annotations

import hashlib
from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np


def fingerprint_state(tree: Any) -> dict[str, Any]:
    digest = hashlib.blake2b(digest_size=16)
    array_count = 0
    element_count = 0
    finite_count = 0
    nan_count = 0
    inf_count = 0
    value_sum = 0.0
    abs_sum = 0.0
    non_finite_leaves: dict[int, dict[str, Any]] = {}
    for leaf_index, (path, leaf) in enumerate(jtu.tree_flatten_with_path(tree)[0]):
        if leaf is None:
            continue
        if isinstance(leaf, jax.Array):
            shards = leaf.addressable_shards
        else:
            shards = (None,)
        for shard in shards:
            array = np.asarray(jax.device_get(leaf if shard is None else shard.data))
            if not np.issubdtype(array.dtype, np.number):
                continue
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            if shard is not None:
                digest.update(repr(shard.index).encode())
                digest.update(np.asarray(leaf.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.view(np.uint8))
            array_count += 1
            element_count += contiguous.size
            if np.issubdtype(contiguous.dtype, np.inexact):
                leaf_finite_count = int(np.isfinite(contiguous).sum())
                leaf_nan_count = int(np.isnan(contiguous).sum())
                leaf_inf_count = int(np.isinf(contiguous).sum())
                finite_count += leaf_finite_count
                nan_count += leaf_nan_count
                inf_count += leaf_inf_count
                if leaf_nan_count or leaf_inf_count:
                    summary = non_finite_leaves.setdefault(
                        leaf_index,
                        {
                            "leaf_index": leaf_index,
                            "path": jtu.keystr(path),
                            "shape": list(leaf.shape)
                            if isinstance(leaf, jax.Array)
                            else list(contiguous.shape),
                            "finite_count": 0,
                            "nan_count": 0,
                            "inf_count": 0,
                        },
                    )
                    summary["finite_count"] += leaf_finite_count
                    summary["nan_count"] += leaf_nan_count
                    summary["inf_count"] += leaf_inf_count
            values = contiguous.astype(np.float64, copy=False)
            value_sum += float(values.sum()) if values.size else 0.0
            abs_sum += float(np.abs(values).sum()) if values.size else 0.0
    if array_count == 0:
        raise ValueError("state fingerprint requires at least one numeric leaf")
    return {
        "array_count": array_count,
        "element_count": element_count,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "non_finite_leaves": [
            non_finite_leaves[index] for index in sorted(non_finite_leaves)
        ],
        "sum": value_sum,
        "abs_sum": abs_sum,
        "fingerprint": digest.hexdigest(),
    }


__all__ = ["fingerprint_state"]
