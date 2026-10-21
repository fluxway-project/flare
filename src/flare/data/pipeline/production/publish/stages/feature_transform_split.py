"""EncodeUDF — PyArrow Expr-based feature encoding stage.

Thin Ray Data ``map_batches`` UDF that delegates the full
``fea_cross → encode → fid_cross → select`` lazy plan to
:class:`flare.data.pipeline.production.builder.expr_plan.ExprPlan`. All heavy state lives in
the pre-built plan; the UDF itself is stateless after construction.
"""

from __future__ import annotations

import time

import cloudpickle
import pyarrow as pa

from flare.data.pipeline.production.builder.expr_plan import ExprPlan
from flare.data.pipeline.production.publish.partitioning import PARTITION_COLUMNS
from flare.data.pipeline.production.publish.stage_profile import (
    StageProfileSnapshot,
    report_stage_profile,
)


class EncodeUDF:
    def __init__(
        self,
        encoded_plan: bytes,
        output_schema: pa.Schema,
        stage_profile_collector=None,
    ) -> None:
        plan = cloudpickle.loads(encoded_plan)
        if not isinstance(plan, ExprPlan):
            raise TypeError("EncodeUDF requires a compiled ExprPlan")
        if not isinstance(output_schema, pa.Schema):
            raise TypeError("EncodeUDF requires a PyArrow output schema")
        self._plan = plan
        self._output_schema = output_schema
        self._expected_output_columns = tuple(self._output_schema.names)
        self._stage_profile_collector = stage_profile_collector

    def __call__(self, batch: pa.Table) -> pa.Table:
        wall_started = time.perf_counter() if self._stage_profile_collector else None
        cpu_started = time.process_time() if self._stage_profile_collector else None
        result = self._plan.transform(batch)
        for name in PARTITION_COLUMNS:
            if name in batch.column_names:
                result = result.append_column(name, batch.column(name))
        expected_columns = self._expected_output_columns + tuple(
            name for name in PARTITION_COLUMNS if name in batch.column_names
        )
        if tuple(result.column_names) != expected_columns:
            raise RuntimeError(
                "EncodeUDF output schema differs from the declared producer schema: "
                f"expected={expected_columns!r} actual={tuple(result.column_names)!r}"
            )
        output_schema = self._output_schema
        for name in PARTITION_COLUMNS:
            if name in batch.column_names:
                output_schema = output_schema.append(pa.field(name, pa.string()))
        # The compiled ExprPlan normally emits the declared Arrow schema
        # exactly.  Avoid an otherwise redundant safe cast on every actor
        # batch, while retaining the strict conversion path for any drift.
        if not result.schema.equals(output_schema):
            result = result.cast(output_schema, safe=True)
        if self._stage_profile_collector is not None:
            if wall_started is None or cpu_started is None:
                raise RuntimeError("encode stage profile timers were not initialized")
            report_stage_profile(
                self._stage_profile_collector,
                StageProfileSnapshot(
                    stage="encode",
                    calls=1,
                    input_rows=batch.num_rows,
                    output_rows=result.num_rows,
                    wall_seconds=time.perf_counter() - wall_started,
                    cpu_seconds=time.process_time() - cpu_started,
                ),
            )
        return result
