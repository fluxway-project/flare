from __future__ import annotations

SparsePooledWidths = tuple[int, ...]
SPARSE_EMBEDDING_OPERATOR_BACKENDS = ("cupy", "cpu_numba", "jax")


def pooled_sparse_widths(target_widths: SparsePooledWidths) -> tuple[int, int]:
    if len(target_widths) != 2:
        raise ValueError(
            "Sparse embedding pooled operations require exactly two target widths, "
            f"got {target_widths}."
        )
    feature_count, max_elements = (int(value) for value in target_widths)
    if feature_count <= 0 or max_elements <= 0:
        raise ValueError(
            "Sparse embedding pooled operations require positive target widths, "
            f"got {target_widths}."
        )
    return feature_count, max_elements


__all__ = [
    "SPARSE_EMBEDDING_OPERATOR_BACKENDS",
    "SparsePooledWidths",
    "pooled_sparse_widths",
]
