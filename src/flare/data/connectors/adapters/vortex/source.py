from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

from flare.data.connectors.adapters.file.source import FileListSourceAdapter
from flare.data.connectors.adapters.vortex.range_reader import read_vortex_range


class VortexSourceAdapter(FileListSourceAdapter):
    def __init__(
        self,
        *,
        files: Sequence[str],
        projection: Sequence[str],
        record_counts: Mapping[str, int],
        storage_options: Mapping[str, Any] | None = None,
    ) -> None:
        counts = dict(record_counts)
        missing = [path for path in files if path not in counts]
        if missing:
            raise ValueError(f"Vortex source is missing record counts: {missing}")
        super().__init__(
            format_id="vortex",
            files=files,
            projection=projection,
            range_reader=partial(
                read_vortex_range,
                storage_options=dict(storage_options or {}),
            ),
            record_counter=counts.__getitem__,
        )
