from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


def read_label_map(path: Path) -> Mapping[str, int]:
    if not path.is_file():
        raise ValueError(f"label map does not exist: {path}")
    result: dict[str, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ValueError(f"invalid label map at {path}:{line_number}")
        index = int(fields[0])
        label = fields[1]
        if label in result or index in result.values():
            raise ValueError(f"duplicate label/index at {path}:{line_number}")
        result[label] = index
    if not result or set(result.values()) != set(range(len(result))):
        raise ValueError(f"label indexes must be contiguous: {path}")
    return MappingProxyType(result)
