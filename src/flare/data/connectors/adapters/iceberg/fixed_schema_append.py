"""Ray Iceberg datasink for one atomic append with a fixed output schema.

This adapter owns Ray's writer-completion callback only.  The publication
protocol, snapshot properties, and result validation belong to ``sink.py``.
"""

from __future__ import annotations

import logging
from typing import Any

import pyarrow as pa
from ray.data import SaveMode
from ray.data._internal.datasource.iceberg_datasink import IcebergDatasink

logger = logging.getLogger(__name__)


def require_fixed_output_schemas(
    block_schemas: list[pa.Schema], expected_schema: pa.Schema
) -> None:
    """Require every Ray writer output block to match the validated schema."""
    # This identity cache is intentionally narrower than schema interning: Ray
    # may reuse an immutable Schema object across blocks, and comparing a
    # 15k-field production schema repeatedly is expensive.  It only avoids a
    # duplicate ``equals`` call for the same live object; every distinct schema
    # remains strictly validated.  ``block_schemas`` keeps the objects alive,
    # so an id cannot be reused during this loop.
    validated: dict[int, bool] = {}
    mismatches = []
    for index, block_schema in enumerate(block_schemas):
        schema_id = id(block_schema)
        matches = validated.get(schema_id)
        if matches is None:
            matches = block_schema.equals(expected_schema)
            validated[schema_id] = matches
        if not matches:
            mismatches.append(index)
    if mismatches:
        raise RuntimeError(
            "Iceberg write block schemas differ from the validated output schema: "
            f"mismatched_blocks={mismatches[:8]}, total_mismatches={len(mismatches)}"
        )


class FixedSchemaAppendIcebergDatasink(IcebergDatasink):
    """Commit exactly one APPEND after fixed-schema validation.

    The publisher has already validated the output against the Iceberg table
    and production does not permit schema evolution.  This callback therefore
    deliberately replaces Ray's general reconciliation path with exact output
    block validation and one append commit.  The private Ray base class and
    protected commit helper make this an explicit, isolated upgrade boundary.
    """

    def __init__(
        self,
        *,
        expected_schema: pa.Schema,
        expected_iceberg_schema: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if self._mode != SaveMode.APPEND:
            raise RuntimeError(
                f"fixed-schema append datasink requires APPEND mode, got {self._mode}"
            )
        self._expected_schema = expected_schema
        self._expected_iceberg_schema = expected_iceberg_schema

    def on_write_complete(self, write_result: Any) -> None:
        import time

        started = time.perf_counter()
        all_data_files = []
        all_schemas: list[pa.Schema] = []
        for write_return in write_result.write_returns:
            if not write_return:
                continue
            if write_return.data_files:
                all_data_files.extend(write_return.data_files)
                all_schemas.extend(write_return.schemas)
        logger.info(
            "[on_write_complete] Fixed-schema validation for %d data files and %d schema blocks ...",
            len(all_data_files),
            len(all_schemas),
        )
        if not all_data_files:
            raise RuntimeError("Iceberg append completed without data files")
        if not all_schemas:
            raise RuntimeError("Iceberg append completed without block schemas")
        require_fixed_output_schemas(all_schemas, self._expected_schema)
        logger.info(
            "[on_write_complete] Fixed-schema validation used %d distinct schema objects",
            len({id(schema) for schema in all_schemas}),
        )
        # Compare canonical Iceberg schemas, not Arrow projections: Iceberg may
        # expose a semantically compatible Arrow ``large_list`` where pipeline
        # output uses ``list``.  The publisher's preflight returned this exact
        # canonical schema; a changed table at commit time must fail rather than
        # silently evolve or reconcile it.
        if self._table.schema() != self._expected_iceberg_schema:
            raise RuntimeError(
                "Iceberg table schema differs from the validated Iceberg schema"
            )
        transaction = self._table.transaction()
        logger.info(
            "[on_write_complete] Starting fixed-schema APPEND commit for %d data files ...",
            len(all_data_files),
        )
        commit_started = time.perf_counter()
        # Protected Ray API by design.  The checks above establish the
        # invariants normally handled by reconciliation, so call only its
        # append/atomic-commit portion and never its schema update path.
        self._append_and_commit(transaction, all_data_files)
        logger.info(
            "[on_write_complete] Fixed-schema append commit complete in %.2fs "
            "(total on_write_complete=%.2fs)",
            time.perf_counter() - commit_started,
            time.perf_counter() - started,
        )
