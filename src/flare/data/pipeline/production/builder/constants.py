from __future__ import annotations

# Consolidated single source of truth for constants and sentinels

# Default sentinels for missing values and padding
DEFAULT_NULL_SENTINEL: int = -3
DEFAULT_FLOAT_NULL_SENTINEL: float = -3.0
DEFAULT_MISSING_INT: int = -1
DEFAULT_MISSING_FLOAT: float = -1.0
DEFAULT_PAD_VALUE: int = -1

# Feature encoding constants
ARRAY_FEATURE_MAX_LENGTH: int = 256
BUCKET_EPSILON: float = 1e-5
NULL_FID_SENTINEL: int = DEFAULT_NULL_SENTINEL

__all__ = [
    "ARRAY_FEATURE_MAX_LENGTH",
    "BUCKET_EPSILON",
    "DEFAULT_FLOAT_NULL_SENTINEL",
    "DEFAULT_MISSING_FLOAT",
    "DEFAULT_MISSING_INT",
    "DEFAULT_NULL_SENTINEL",
    "DEFAULT_PAD_VALUE",
    "NULL_FID_SENTINEL",
]
