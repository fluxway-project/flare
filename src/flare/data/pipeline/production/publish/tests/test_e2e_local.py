"""End-to-end local validation for the ETL pipeline.

Two operating modes:

* ``pytest src/flare/data/pipeline/production/publish/tests/test_e2e_local.py``
    Fast correctness validation — small batches, assertion-driven.

* ``python -m flare.data.pipeline.production.publish.tests.test_e2e_local [--rows N --iters K --ray]``
    Throughput benchmarking — large batches, prints rows/sec.

The script intentionally bypasses the Cython ParseUDF stage and feeds an
``ExprPlan`` directly with synthetic Arrow tables shaped like the parser's
output, so the encode / cross logic can be exercised in isolation.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from flare.data.pipeline.production.builder.cross.fea_ops import (
    HitCountCrossExpr,
    SubCrossExpr,
)
from flare.data.pipeline.production.builder.cross.fid_ops import FidCrossExpr
from flare.data.pipeline.production.builder.encoding.dense import (
    AutoCategoryEncoderExpr,
    BucketEncoderExpr,
    BucketPartition,
    CategoryEncoderExpr,
    CategoryLookupTable,
    RawFloatEncoderExpr,
)
from flare.data.pipeline.production.builder.encoding.protocol import (
    SLOT_BITS_U64,
    encoded_missing_for,
    sparse_fid_i64,
)
from flare.data.pipeline.production.builder.encoding.sparse_id import AsItIsEncoderExpr
from flare.data.pipeline.production.builder.expr_plan import (
    ExprPlan,
    _CrossExecutionContext,
)
from flare.data.schema.feature_catalog import FeatureOutputKind

# ---------------------------------------------------------------------------
# Plan / data construction helpers
# ---------------------------------------------------------------------------

# Slot assignments are distinct so the int64 FID bit layout can be verified.
SLOT_AGE = 2
SLOT_CITY = 1
SLOT_ENTITY = 3
SLOT_SALARY = 4
SLOT_AGE_DIFF = 5
SLOT_PROFILE = 6

BUCKET_BOUNDS = [10.0, 20.0, 30.0, 40.0, 50.0]
CATEGORY_MAPPING = {100: 1, 200: 2, 300: 3}
CATEGORY_DEFAULT = 0


def make_synthetic_batch(batch_size: int, *, seed: int = 0) -> pa.Table:
    """Generate an Arrow table matching the schema expected by the demo plan.

    Includes one ``unused_col`` to exercise projection pruning.
    """
    rng = np.random.default_rng(seed)
    return pa.table(
        {
            "age": rng.uniform(18, 60, batch_size).astype(np.float32),
            "exp_years": rng.uniform(0, 30, batch_size).astype(np.float32),
            "city": rng.choice(
                np.array([100, 200, 300, 999], dtype=np.int32), batch_size
            ),
            "entity_id": rng.integers(1, 10_000_000, batch_size, dtype=np.int64),
            "salary": rng.uniform(3000, 50000, batch_size).astype(np.float32),
            "profile_level": rng.integers(0, 1000, batch_size, dtype=np.int64),
            "unused_col": np.zeros(batch_size, dtype=np.int64),
        }
    )


def build_plan() -> ExprPlan:
    """Construct a representative plan exercising every encoder family + a cross."""
    bucket_enc = BucketEncoderExpr(partition=BucketPartition.from_bounds(BUCKET_BOUNDS))
    category_enc = CategoryEncoderExpr(
        table=CategoryLookupTable.from_mapping(
            CATEGORY_MAPPING, default_index=CATEGORY_DEFAULT
        ),
        default_index=CATEGORY_DEFAULT,
    )
    as_it_is_enc = AsItIsEncoderExpr()
    float_enc = RawFloatEncoderExpr()
    auto_enc = AutoCategoryEncoderExpr()

    sub_cross = SubCrossExpr(name="age_diff", dep_names=["age", "exp_years"])

    encode_specs: list[tuple[str, int, int, Any]] = [
        ("age", SLOT_AGE, 1, bucket_enc),
        ("city", SLOT_CITY, 1, category_enc),
        ("entity_id", SLOT_ENTITY, 1, as_it_is_enc),
        ("salary", SLOT_SALARY, 1, float_enc),
        ("age_diff", SLOT_AGE_DIFF, 1, float_enc),
        ("profile_level", SLOT_PROFILE, 1, auto_enc),
    ]
    # age_diff is produced by the fea_cross above and is passed straight through
    # to the output (DENSE_FLOAT-style) — exercises the fea_cross plumbing without
    # re-encoding the cross output.
    output_cols = [
        f"fid_{SLOT_AGE}",
        f"fid_{SLOT_CITY}",
        f"fid_{SLOT_ENTITY}",
        f"dense_{SLOT_SALARY}",
        f"dense_{SLOT_AGE_DIFF}",
        f"fid_{SLOT_PROFILE}",
    ]

    return ExprPlan(
        fea_cross_ops=[sub_cross],  # pyrefly: ignore[bad-argument-type]
        encode_specs=encode_specs,
        fid_cross_ops=[],
        output_cols=output_cols,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_plan() -> ExprPlan:
    return build_plan()


@pytest.fixture
def synthetic_batch() -> pa.Table:
    return make_synthetic_batch(1024, seed=42)


# ---------------------------------------------------------------------------
# Correctness — output schema & dtypes
# ---------------------------------------------------------------------------


class TestExprPlanCorrectness:
    def test_output_schema(
        self, simple_plan: ExprPlan, synthetic_batch: pa.Table
    ) -> None:
        result = simple_plan.transform(synthetic_batch)
        assert list(result.column_names) == simple_plan._output_cols

    def test_u32_fid_range(
        self, simple_plan: ExprPlan, synthetic_batch: pa.Table
    ) -> None:
        result = simple_plan.transform(synthetic_batch)
        max_bucket = len(
            BUCKET_BOUNDS
        )  # cut produces len(breaks)+1 buckets, idx 0..len
        age_arr = result.column(f"fid_{SLOT_AGE}").to_numpy(zero_copy_only=False)
        assert age_arr.dtype == np.int32, (
            f"age_fid dtype expected int32, got {age_arr.dtype}"
        )
        assert int(age_arr.min()) >= 0
        assert int(age_arr.max()) <= max_bucket

        city_arr = result.column(f"fid_{SLOT_CITY}").to_numpy(zero_copy_only=False)
        assert city_arr.dtype == np.int32
        assert {int(v) for v in city_arr} <= {0, 1, 2, 3}

    def test_u64_fid_slot(
        self, simple_plan: ExprPlan, synthetic_batch: pa.Table
    ) -> None:
        result = simple_plan.transform(synthetic_batch)
        entity_arr = result.column(f"fid_{SLOT_ENTITY}").to_numpy(zero_copy_only=False)
        assert entity_arr.dtype == np.int64
        slots = entity_arr.view(np.uint64) >> SLOT_BITS_U64
        assert (slots == SLOT_ENTITY).all(), (
            f"entity_id_fid slot mismatch: unique slots={set(slots.tolist())}"
        )

        profile_arr = result.column(f"fid_{SLOT_PROFILE}").to_numpy(
            zero_copy_only=False
        )
        assert profile_arr.dtype == np.int64
        assert ((profile_arr.view(np.uint64) >> SLOT_BITS_U64) == SLOT_PROFILE).all()

    def test_sparse_i64_list_uses_one_missing_id_without_width_padding(self) -> None:
        recent_slot = 3269
        auto_slot = 3270
        plan = ExprPlan(
            fea_cross_ops=[],
            encode_specs=[
                ("recent", recent_slot, 30, AsItIsEncoderExpr()),
                ("auto_recent", auto_slot, 30, AutoCategoryEncoderExpr()),
            ],
            fid_cross_ops=[],
            output_cols=[f"fid_{recent_slot}", f"fid_{auto_slot}"],
        )
        batch = pa.table(
            {
                "recent": pa.array([[], [123, 456], []], type=pa.list_(pa.int64())),
                "auto_recent": pa.array(
                    [[789], [], [321, 654]], type=pa.list_(pa.int64())
                ),
            }
        )

        result = plan.transform(batch)
        recent = result.column(f"fid_{recent_slot}").combine_chunks()
        auto_recent = result.column(f"fid_{auto_slot}").combine_chunks()

        assert pc.list_value_length(recent).to_pylist() == [  # pyrefly: ignore[missing-attribute]
            1,
            2,
            1,
        ]
        assert recent.values.to_pylist() == [
            encoded_missing_for(FeatureOutputKind.SPARSE_I64_FIDS, recent_slot),
            sparse_fid_i64(recent_slot) | 123,
            sparse_fid_i64(recent_slot) | 456,
            encoded_missing_for(FeatureOutputKind.SPARSE_I64_FIDS, recent_slot),
        ]
        assert pc.list_value_length(auto_recent).to_pylist() == [  # pyrefly: ignore[missing-attribute]
            1,
            1,
            2,
        ]
        assert auto_recent.values.to_pylist() == [
            sparse_fid_i64(auto_slot) | 789,
            encoded_missing_for(FeatureOutputKind.SPARSE_I64_FIDS, auto_slot),
            sparse_fid_i64(auto_slot) | 321,
            sparse_fid_i64(auto_slot) | 654,
        ]

    def test_float_no_nan(
        self, simple_plan: ExprPlan, synthetic_batch: pa.Table
    ) -> None:
        result = simple_plan.transform(synthetic_batch)
        salary_arr = result.column(f"dense_{SLOT_SALARY}").to_numpy(
            zero_copy_only=False
        )
        assert salary_arr.dtype == np.float32
        assert np.isfinite(salary_arr).all()
        # age_diff is produced by SubCrossExpr (cast to Float32) — also must be finite.
        diff_arr = result.column(f"dense_{SLOT_AGE_DIFF}").to_numpy(
            zero_copy_only=False
        )
        assert diff_arr.dtype == np.float32
        assert np.isfinite(diff_arr).all()

    def test_unused_col_excluded(
        self, simple_plan: ExprPlan, synthetic_batch: pa.Table
    ) -> None:
        result = simple_plan.transform(synthetic_batch)
        assert "unused_col" not in result.column_names

    def test_projection_pruning(self, simple_plan: ExprPlan) -> None:
        assert "unused_col" not in simple_plan._input_cols
        # Sanity: every name actually consumed by the plan IS projected.
        for required in (
            "age",
            "exp_years",
            "city",
            "entity_id",
            "salary",
            "profile_level",
        ):
            assert required in simple_plan._input_cols

    def test_encoder_cannot_silently_omit_declared_output(self) -> None:
        class EmptyEncoder(AsItIsEncoderExpr):
            def table_compute(self, table, col_name, slot, length):
                return None

        plan = ExprPlan(
            fea_cross_ops=[],
            encode_specs=[("value", 1, 1, EmptyEncoder())],
            fid_cross_ops=[],
            output_cols=["fid_1"],
        )

        with pytest.raises(RuntimeError, match="feature code 1 produced no output"):
            plan.transform(pa.table({"value": [1]}))


# ---------------------------------------------------------------------------
# Correctness — fid_cross int64 bit pattern
# ---------------------------------------------------------------------------


class TestFidCrossDtype:
    def test_fid_cross_int64_bit_pattern(self) -> None:
        plan = ExprPlan(
            fea_cross_ops=[],
            encode_specs=[
                ("entity_id", SLOT_ENTITY, 1, AsItIsEncoderExpr()),
                ("profile_level", SLOT_PROFILE, 1, AutoCategoryEncoderExpr()),
            ],
            fid_cross_ops=[  # pyrefly: ignore[bad-argument-type]
                FidCrossExpr(
                    name="fid_7",
                    dep_names=[f"fid_{SLOT_ENTITY}", f"fid_{SLOT_PROFILE}"],
                    bit_sizes=[24, 24],
                    dep_widths=[1, 1],
                    slot=7,
                )
            ],
            output_cols=[f"fid_{SLOT_ENTITY}", f"fid_{SLOT_PROFILE}", "fid_7"],
        )
        batch = make_synthetic_batch(256, seed=1)
        result = plan.transform(batch)
        for col in (f"fid_{SLOT_ENTITY}", f"fid_{SLOT_PROFILE}", "fid_7"):
            arr = result.column(col).to_numpy(zero_copy_only=False)
            assert arr.dtype == np.int64, f"{col} should be Int64, got {arr.dtype}"

    def test_list_fid_cross_preserves_empty_input(self) -> None:
        left_slot = 3269
        right_slot = 125
        cross_slot = 60008
        plan = ExprPlan(
            fea_cross_ops=[],
            encode_specs=[
                ("left", left_slot, 30, AsItIsEncoderExpr()),
                ("right", right_slot, 30, AsItIsEncoderExpr()),
            ],
            fid_cross_ops=[  # pyrefly: ignore[bad-argument-type]
                FidCrossExpr(
                    name=f"fid_{cross_slot}",
                    dep_names=[f"fid_{left_slot}", f"fid_{right_slot}"],
                    bit_sizes=[24, 24],
                    dep_widths=[30, 30],
                    slot=cross_slot,
                )
            ],
            output_cols=[f"fid_{left_slot}", f"fid_{right_slot}", f"fid_{cross_slot}"],
        )
        batch = pa.table(
            {
                "left": pa.array([[], [11]], type=pa.list_(pa.int64())),
                "right": pa.array([[], [22]], type=pa.list_(pa.int64())),
            }
        )

        result = plan.transform(batch)
        crossed = result.column(f"fid_{cross_slot}").combine_chunks()

        assert pc.list_value_length(crossed).to_pylist() == [  # pyrefly: ignore[missing-attribute]
            0,
            1,
        ]
        crossed_bits = crossed.values.to_numpy().view(np.uint64)
        assert crossed_bits[0] >> np.uint64(SLOT_BITS_U64) == cross_slot


class TestHitCountCrossCaching:
    def test_repeated_hitcount_outputs_match_direct_compute(self) -> None:
        batch = pa.table(
            {
                "a": pa.array([[1, 2, 3], [4, 5], [6, 7, 8], [9]]),
                "b": pa.array([[3, 2, 1], [5, 6], [8, 7, 6], [10]]),
                "label_primary": np.zeros(4, dtype=np.int32),
                "label_secondary": np.zeros(4, dtype=np.int32),
            }
        )
        ops = [
            HitCountCrossExpr(
                name=f"hit_{i}",
                dep_names=["a", "b"],
                list_left=True,
                list_right=True,
                in_dtype1=pa.int64(),
                in_dtype2=pa.int64(),
                out_dtype=pa.float32(),
            )
            for i in range(12)
        ]
        plan = ExprPlan(
            fea_cross_ops=ops,  # pyrefly: ignore[bad-argument-type]
            encode_specs=[],
            fid_cross_ops=[],
            output_cols=[op.name for op in ops],
        )
        result = plan.transform(batch)
        expected = ops[0].table_compute(_CrossExecutionContext(batch))
        assert expected is not None
        for op in ops:
            assert (
                result.column(op.name)
                .combine_chunks()
                .equals(expected.combine_chunks())
            )


# ---------------------------------------------------------------------------
# Throughput benchmarks (command-line mode)
# ---------------------------------------------------------------------------


def bench_single_batch(plan: ExprPlan, batch: pa.Table, n_iters: int = 100) -> dict:
    """Hot-loop ``plan.transform`` and report throughput."""
    plan.transform(batch)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        plan.transform(batch)
    elapsed = time.perf_counter() - t0

    batch_size = batch.num_rows
    total_rows = batch_size * n_iters
    rows_per_sec = total_rows / elapsed if elapsed > 0 else float("inf")
    avg_ms = elapsed / n_iters * 1000.0

    print(f"[bench:single]  batch_size    = {batch_size}")
    print(f"[bench:single]  iterations    = {n_iters}")
    print(f"[bench:single]  total_rows    = {total_rows:,}")
    print(f"[bench:single]  elapsed       = {elapsed:.3f} s")
    print(f"[bench:single]  throughput    = {rows_per_sec:,.0f} rows/sec")
    print(f"[bench:single]  avg_latency   = {avg_ms:.3f} ms/batch")
    return {
        "batch_size": batch_size,
        "iterations": n_iters,
        "total_rows": total_rows,
        "elapsed_sec": elapsed,
        "rows_per_sec": rows_per_sec,
        "avg_latency_ms": avg_ms,
    }


def bench_ray_pipeline(
    *, batch_size: int, n_batches: int, num_cpus: int = 4, num_gpus: int = 1
) -> dict:
    """End-to-end throughput via Ray Data ``map_batches`` (optional)."""
    import ray
    import ray.data

    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)

    use_gpu = num_gpus > 0

    seed_base = 0xC0FFEE
    arrow_batches = [
        make_synthetic_batch(batch_size, seed=seed_base + i) for i in range(n_batches)
    ]
    total_rows = batch_size * n_batches

    class _Encoder:
        def __init__(self) -> None:
            self._plan = build_plan()

        def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            tbl = pa.table({k: v for k, v in batch.items()})
            out = self._plan.transform(tbl)
            return {
                name: out.column(name).to_numpy(zero_copy_only=False)
                for name in out.column_names
            }

    ds = ray.data.from_arrow(arrow_batches)

    t0 = time.perf_counter()
    materialised = ds.map_batches(
        _Encoder,
        batch_format="numpy",
        batch_size=batch_size,
        concurrency=1,
        num_gpus=1 if use_gpu else 0,
    ).materialize()
    elapsed = time.perf_counter() - t0

    out_rows = materialised.count()
    rows_per_sec = out_rows / elapsed if elapsed > 0 else float("inf")

    print(f"[bench:ray]     requested_gpus = {num_gpus}")
    print(f"[bench:ray]     n_batches     = {n_batches}")
    print(f"[bench:ray]     batch_size    = {batch_size}")
    print(f"[bench:ray]     input_rows    = {total_rows:,}")
    print(f"[bench:ray]     output_rows   = {out_rows:,}")
    print(f"[bench:ray]     elapsed       = {elapsed:.3f} s")
    print(f"[bench:ray]     throughput    = {rows_per_sec:,.0f} rows/sec")

    # Don't shutdown ray here — caller may want to reuse the cluster.
    return {
        "requested_gpus": num_gpus,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "input_rows": total_rows,
        "output_rows": out_rows,
        "elapsed_sec": elapsed,
        "rows_per_sec": rows_per_sec,
    }


# ---------------------------------------------------------------------------
# CLI entry — `python -m flare.data.pipeline.production.publish.tests.test_e2e_local`
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Throughput benchmark for the ETL pipeline."
    )
    parser.add_argument(
        "--rows", type=int, default=8192, help="Batch size for single-batch bench."
    )
    parser.add_argument(
        "--iters", type=int, default=200, help="Iteration count for single-batch bench."
    )
    parser.add_argument(
        "--ray", action="store_true", help="Also run Ray end-to-end bench."
    )
    parser.add_argument(
        "--ray-batches", type=int, default=8, help="Number of batches for Ray bench."
    )
    parser.add_argument(
        "--ray-batch-size", type=int, default=8192, help="Batch size for Ray bench."
    )
    args = parser.parse_args(argv)

    plan = build_plan()
    batch = make_synthetic_batch(args.rows, seed=0xBEEF)
    bench_single_batch(plan, batch, n_iters=args.iters)

    if args.ray:
        bench_ray_pipeline(batch_size=args.ray_batch_size, n_batches=args.ray_batches)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
