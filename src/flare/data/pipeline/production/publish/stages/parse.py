"""flare.data.pipeline.production.publish.stages.parse — ParseUDF for Ray Data map_batches.

Stateful CPU UDF for the parse stage of the stage-split pipeline.

Stage layout::

    HdfsZstdDatasource         (IO + zstd decompress, 1 task per file)
        → {"line": large_string}            [chunk_rows rows / Arrow block]
        map_batches(ParseUDF)      (line-parse, large CPU actor pool)
        → pa.Table (raw_arrow_schema)        [zero-copy ndarray buffers]
    map_batches(EncodeUDF)     (cross + encode, GPU/CPU actor pool)
        → pa.Table (vortex_schema)
    DatasetTerminal

Decoupling line-parse from the Datasource lets Ray Data scale the parse
pool independently of file count — IO is throttled by ``len(file_paths)``
ReadTasks, but parsing scales with ``StageTopologyPlanner.n_parse``.

Zero-copy contract (vs. the historical pickle blob): ParseUDF emits one
Arrow column per base feature + 2 meta columns. Each column reuses the
ndarray buffer preallocated by ``RowParser.parse_lines`` — no pickle, no
column copy on either side of the actor boundary.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pyarrow as pa

from flare.data.pipeline.production.builder.row_parser import FeatureBuffers
from flare.data.pipeline.production.publish.partitioning import PARTITION_COLUMNS
from flare.data.pipeline.production.publish.stage_profile import (
    StageProfileSnapshot,
    report_stage_profile,
)

logger = logging.getLogger(__name__)


def _raw_features_to_arrow(
    raw_features: FeatureBuffers,
    meta_ints: dict[str, np.ndarray],
    schema: pa.Schema,
    batch_size: int,
) -> pa.Table:
    """Build a pa.Table from preallocated ndarray buffers — zero-copy.

    For each schema field:
      - scalar (int64/float32) → ``pa.array(ndarray, type=...)`` reuses the
        ndarray buffer.
      - fixed_size_list → ``pa.FixedSizeListArray.from_arrays`` with a
        flat view (``ndarray.ravel()`` is a view on C-contiguous 2D arrays).
    """
    arrays: list[pa.Array] = []
    for fld in schema:
        name = fld.name
        if name in raw_features:
            buf = raw_features[name]
        else:
            buf = meta_ints[name]

        if pa.types.is_list(fld.type) or pa.types.is_large_list(fld.type):
            value_type = fld.type.value_type
            flat_buf, offsets_buf = buf
            offsets_buf.resize(batch_size + 1, refcheck=False)
            total_elements = offsets_buf[-1]
            flat_buf.resize(total_elements, refcheck=False)
            arrays.append(
                pa.ListArray.from_arrays(
                    pa.array(offsets_buf),
                    pa.array(flat_buf, type=value_type),
                )
            )
        else:
            buf.resize(batch_size, refcheck=False)
            if name in raw_features:
                validity = raw_features.scalar_validity[name]
                validity.resize(batch_size, refcheck=False)
                arrays.append(pa.array(buf, mask=validity == 0, type=fld.type))
            else:
                arrays.append(pa.array(buf, type=fld.type))
    return pa.table(arrays, schema=schema)


class ParseUDF:
    """Stateful UDF for ``ray.data.map_batches`` — text lines → Arrow features.

    Wraps :class:`flare.data.pipeline.production.builder.row_parser.RowParser`.

    Input batch:
        ``pa.Table`` with a single ``line`` column (large_string), one row
        per text line. Each Arrow block from the Datasource maps 1:1 to one
        ``__call__`` invocation.

    Output batch:
        ``pa.Table`` with one column per base feature plus explicitly configured
        direct and metadata columns. Schema is fixed by
        ``RowParser.raw_arrow_schema()``; columns reuse the ndarray buffers
        from ``RowParser.parse_lines`` so the EncodeUDF can take zero-copy
        ndarray views back without any pickle round-trip.
    """

    def __init__(
        self,
        parser_spec,
        metadata_maps: dict[str, dict[str, int]],
        stage_profile_collector=None,
    ):
        from flare.data.pipeline.production.builder.row_parser import RowParser
        from flare.data.pipeline.production.processing.parser_spec import ParserSpec

        if not isinstance(parser_spec, ParserSpec):
            raise TypeError("ParseUDF requires a ParserSpec")
        self._parser: RowParser | None = RowParser(parser_spec, metadata_maps)
        self._arrow_schema = parser_spec.raw_schema
        self._total_lines = 0
        self._total_blocks = 0
        self._total_time = 0.0
        self._stage_profile_collector = stage_profile_collector
        logger.info("ParseUDF initialized (pid=%d)", os.getpid())

    def __del__(self) -> None:
        self._parser = None

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Parse one block of raw lines into a multi-column Arrow Table.

        Uses the zero-copy Arrow buffer path: the LargeStringArray's
        Arrow strings are decoded once by the parser and converted into the
        explicitly configured raw schema.
        """
        n = batch.num_rows
        if n == 0:
            return self._arrow_schema.empty_table()
        parser = self._parser
        if parser is None:
            raise RuntimeError("ParseUDF is closed")

        # Ensure single-chunk for contiguous buffer layout.
        col = batch.column("line")
        if col.num_chunks > 1:
            col = col.combine_chunks()
        else:
            col = col.chunk(0)

        t0 = time.perf_counter()
        cpu_started = time.process_time() if self._stage_profile_collector else None
        raw_features, meta_ints, batch_size = parser.parse_arrow(col)
        elapsed = time.perf_counter() - t0

        if batch_size == 0:
            return self._arrow_schema.empty_table()

        out = _raw_features_to_arrow(
            raw_features, meta_ints, self._arrow_schema, batch_size
        )

        for name in PARTITION_COLUMNS:
            if name in batch.column_names:
                out = out.append_column(name, batch.column(name))

        if self._stage_profile_collector is not None:
            if cpu_started is None:
                raise RuntimeError("parse stage profile CPU timer was not initialized")
            report_stage_profile(
                self._stage_profile_collector,
                StageProfileSnapshot(
                    stage="parse",
                    calls=1,
                    input_rows=n,
                    output_rows=out.num_rows,
                    wall_seconds=time.perf_counter() - t0,
                    cpu_seconds=time.process_time() - cpu_started,
                ),
            )

        self._total_lines += batch_size
        self._total_blocks += 1
        self._total_time += elapsed

        if self._total_blocks % 50 == 0:
            logger.info(
                "ParseUDF stats: %d blocks, %d lines, %.0f lines/s",
                self._total_blocks,
                self._total_lines,
                self._total_lines / max(self._total_time, 0.001),
            )

        return out
