from __future__ import annotations

import importlib
from types import ModuleType


def load_native_scan_extension() -> ModuleType:
    try:
        return importlib.import_module(
            "flare.data.pipeline.production.processing.native._scan"
        )
    except ImportError as exc:
        raise RuntimeError(
            "Native scan extension flare.data.pipeline.production.processing.native._scan is not available. "
            "Please build extensions with `pip install -e .` or `python setup.py build_ext --inplace`."
        ) from exc
