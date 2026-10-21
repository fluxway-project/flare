from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pyarrow as pa

from flare.data.contracts.sink import SinkResult

if TYPE_CHECKING:
    import ray.data


class DatasetTerminal(Protocol):
    """Publish one completed dataset to its output destination."""

    def finalize(
        self, dataset: ray.data.Dataset, *, schema: pa.Schema
    ) -> SinkResult: ...
