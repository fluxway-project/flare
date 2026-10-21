from __future__ import annotations

import numpy as np
import pytest
from fluxway.common.types.tensor import TransportTensor

from flare.runtime.sparse_precision import (
    prepare_sparse_native_gradient,
    require_finite_sparse_values,
    require_key_dtype,
    require_sparse_key_dtype,
)
from flare.training.components.sparse_table.contracts import SparseTableSpec
from recipes.recsys.ranking.schema.materialization import (
    materialize_recsys_sparse_inputs,
)


def test_sparse_worker_rejects_nonfinite_values_before_continuing() -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        require_finite_sparse_values(
            np.asarray([1.0, np.nan], dtype=np.float16),
            "float16",
            boundary="test",
        )


@pytest.mark.parametrize(
    "dtype",
    ["float16", "float32"],
)
def test_sparse_worker_preserves_finite_values_and_rejects_ieee_values(
    dtype: str,
) -> None:
    source = np.asarray([1.5, -0.0], dtype=dtype)
    assert require_finite_sparse_values(source, dtype, boundary="test") is source
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(
            FloatingPointError, match="non-finite sparse values at test"
        ):
            require_finite_sparse_values(
                np.asarray([invalid], dtype=dtype), dtype, boundary="test"
            )


def test_sparse_worker_accepts_bfloat16_transport_and_rejects_nonfinite_values() -> (
    None
):
    import ml_dtypes

    source = np.asarray([1.5, -0.0], dtype=ml_dtypes.bfloat16)
    assert require_finite_sparse_values(source, "bfloat16", boundary="test") is source
    with pytest.raises(FloatingPointError, match="non-finite sparse values at test"):
        require_finite_sparse_values(
            np.asarray([np.nan], dtype=ml_dtypes.bfloat16),
            "bfloat16",
            boundary="test",
        )


def test_sparse_worker_rejects_jax_nonfinite_values() -> None:
    import jax.numpy as jnp

    source = jnp.asarray([jnp.nan, jnp.inf, -jnp.inf], dtype=jnp.float16)
    with pytest.raises(FloatingPointError, match="non-finite"):
        require_finite_sparse_values(source, "float16", boundary="test")


def test_sparse_worker_rejects_dtype_contract_violations() -> None:
    with pytest.raises(ValueError, match="float16, bfloat16, or float32"):
        require_finite_sparse_values(
            np.ones(2, dtype=np.float16), "float64", boundary="test"
        )
    with pytest.raises(TypeError, match="dtype mismatch"):
        require_finite_sparse_values(
            np.ones(2, dtype=np.float32), "float16", boundary="test"
        )


@pytest.mark.parametrize("key_dtype", ["int32", "uint32", "int64", "uint64"])
def test_sparse_worker_uses_owner_key_dtype(key_dtype: str) -> None:
    owner_dtype = key_dtype
    assert require_key_dtype(owner_dtype) == key_dtype
    keys = np.asarray([1], dtype=np.dtype(key_dtype))
    assert require_sparse_key_dtype(keys, owner_dtype, boundary="test") is keys


def test_sparse_worker_rejects_key_dtype_mismatch() -> None:
    with pytest.raises(TypeError, match="sparse key dtype mismatch at test"):
        require_sparse_key_dtype(
            np.asarray([1], dtype=np.int64), "uint64", boundary="test"
        )


def test_sparse_native_gradient_cast_is_explicit_and_finite() -> None:
    source = np.asarray([[np.nan, np.inf]], dtype=np.float32)
    with pytest.raises(FloatingPointError, match="sparse_native_gradient"):
        prepare_sparse_native_gradient(
            source, transport_dtype="float32", native_dtype="float32",
            validate_values=True,
        )
    with pytest.raises(TypeError, match="transport dtype mismatch"):
        prepare_sparse_native_gradient(
            np.ones(2, dtype=np.float16),
            transport_dtype="float32",
            native_dtype="float32",
        )


def test_sparse_native_gradient_rejects_fp16_transport() -> None:
    with pytest.raises(ValueError, match=r"must be one of \['bfloat16', 'float32'\]"):
        prepare_sparse_native_gradient(
            np.ones(2, dtype=np.float16),
            transport_dtype="float16",
            native_dtype="float32",
        )


def test_sparse_native_gradient_promotes_bfloat16_transport_to_fp32() -> None:
    import ml_dtypes

    source = np.asarray([[1.5, -2.0]], dtype=ml_dtypes.bfloat16)
    result = prepare_sparse_native_gradient(
        source, transport_dtype="bfloat16", native_dtype="float32"
    )
    assert result.dtype == np.dtype("float32")
    np.testing.assert_allclose(result, np.asarray([[1.5, -2.0]], dtype=np.float32))


def test_sparse_table_spec_rejects_fp16_gradient_contract() -> None:
    with pytest.raises(ValueError, match="float32"):
        SparseTableSpec(
            name="group",
            embedding_dimensions=(128,),
            shards=1,
            value_dtype="float16",
            gradient_transport_dtype="float16",
            gradient_dtype="float32",
            optimizer_dtype="float32",
        )


def test_sparse_table_spec_accepts_bfloat16_transport_and_fp32_native() -> None:
    spec = SparseTableSpec(
        name="group",
        embedding_dimensions=(128,),
        shards=1,
        value_dtype="float16",
        gradient_transport_dtype="bfloat16",
        gradient_dtype="float32",
        optimizer_dtype="float32",
    )
    assert spec.gradient_transport_dtype == "bfloat16"


def test_materialization_rejects_pooled_nonfinite_values_before_model_boundary() -> (
    None
):
    class _SparseOperator:
        value_dtype = "float16"
        inverse_host_dtype = np.int32

        @staticmethod
        def array(value):
            return np.asarray(value)

        @staticmethod
        def inverse(value):
            return np.asarray(value, dtype=np.int32)

        @staticmethod
        def gather_pooled(*_args):
            return np.asarray([[[np.nan, np.inf]], [[1.0, -np.inf]]], dtype=np.float16)

    embedding = TransportTensor(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
        global_shape=(2, 2),
    )
    inverse = TransportTensor(
        np.asarray([[0], [1]], dtype=np.int32), global_shape=(2, 1)
    )
    labels = TransportTensor(np.asarray([1, 0], dtype=np.int32), global_shape=(2,))
    valid_count = TransportTensor(np.asarray([2], dtype=np.int32), global_shape=(1,))

    with pytest.raises(FloatingPointError, match="model_gather:group"):
        materialize_recsys_sparse_inputs(
            kwargs={
                "labels": labels,
                "samples_valid_count": valid_count,
                "emb_group": embedding,
                "inv_group": inverse,
            },
            sparse_groups=("group",),
            side_info_groups=(),
            sparse_group_widths={"group": (1,)},
            sparse_op=_SparseOperator(),
            world_size=1,
            execution_tracer=None,
            rank=0,
            spmd_id="test",
            operator_owner=object(),
            validate_tensor_values=True,
        )
