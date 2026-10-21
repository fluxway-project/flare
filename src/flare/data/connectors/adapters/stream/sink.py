from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import pyarrow as pa

from flare.data.contracts.sink import SinkResult
from flare.data.pipeline.production.publication.contract import PublicationProtocol

if TYPE_CHECKING:
    import ray.data

logger = logging.getLogger(__name__)
_INFLIGHT_FORMATS = ("arrow", "numpy")


class RayDataStreamDrainer:
    @property
    def publication_protocol(self) -> PublicationProtocol:
        return PublicationProtocol.NON_TRANSACTIONAL

    def __init__(
        self,
        output_format: Literal["arrow", "numpy"] = "arrow",
        prefetch_batches: int = 2,
    ):
        if output_format not in _INFLIGHT_FORMATS:
            raise ValueError(
                f"output_format must be one of {_INFLIGHT_FORMATS}, got {output_format!r}"
            )
        self._format = output_format
        self._prefetch = prefetch_batches
        self.dataset: ray.data.Dataset | None = None

    def finalize(
        self,
        dataset: ray.data.Dataset,
        *,
        schema: pa.Schema,
    ) -> SinkResult:
        total_rows = 0
        for batch in dataset.iter_batches(
            batch_size=None, batch_format="pyarrow", prefetch_batches=self._prefetch
        ):
            if not isinstance(batch, pa.Table):
                raise TypeError("RayDataStreamDrainer requires PyArrow Table batches")
            total_rows += batch.num_rows
            del batch
        logger.info(
            "RayDataStreamDrainer[%s]: drained %d rows (streaming; object store pressure O(%d blocks))",
            self._format,
            total_rows,
            self._prefetch + 1,
        )
        return SinkResult(total_rows=total_rows)
