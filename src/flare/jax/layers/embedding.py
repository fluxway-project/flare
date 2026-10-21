from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from flax import nnx

from flare.jax.layers.initialization import require_rngs
from flare.jax.layers.state import ConstantBuffer
from flare.jax.roles import Fp32Param, cast_floating_tree


class CategoricalEmbedding(nnx.Module):
    """Unified categorical and bucketized embedding layer for side information."""

    def __init__(
        self,
        cate_feature_metas: dict[str, tuple[int, int]],
        side_info_layout: dict[str, tuple[int, int]] | None = None,
        pooling: str = "sum",
        rngs: nnx.Rngs | None = None,
    ):
        rngs = require_rngs(rngs)
        if pooling not in ("sum", "mean"):
            raise ValueError(f"Unsupported pooling method: {pooling}")
        if side_info_layout is None:
            raise ValueError(
                "side_info_layout must be provided for vectorized gathering."
            )

        self.pooling = pooling
        self.cate_feature_metas = cate_feature_metas

        total_vocab_per_dim: dict[int, int] = {}
        vocab_offsets: dict[str, int] = {}

        grouped_features = {}

        for k, (vocab_size, dim) in cate_feature_metas.items():
            padded_vocab = vocab_size + 2
            if dim not in total_vocab_per_dim:
                total_vocab_per_dim[dim] = 0

            vocab_offsets[k] = total_vocab_per_dim[dim]
            total_vocab_per_dim[dim] += padded_vocab

            if k not in side_info_layout:
                raise ValueError(f"Feature {k!r} not found in side_info_layout")

            _offset, fea_len = side_info_layout[k]
            key = (dim, fea_len)
            if key not in grouped_features:
                grouped_features[key] = []
            grouped_features[key].append(k)

        for dim, total_vocab in total_vocab_per_dim.items():
            setattr(
                self,
                f"emb_{dim}",
                Fp32Param(
                    jax.random.normal(
                        rngs.params(), (total_vocab, dim), dtype=jnp.float32
                    )
                ),
            )

        self.feature_order_per_dim: dict[int, list[str]] = {
            dim: [] for dim in total_vocab_per_dim
        }

        gather_meta: list[dict] = []
        gather_indices: list[ConstantBuffer] = []
        gather_v_offsets: list[ConstantBuffer] = []

        for (dim, fea_len), fea_keys in grouped_features.items():
            offsets = [side_info_layout[k][0] for k in fea_keys]
            indices_py: list[list[int]] = [
                [off + i for i in range(fea_len)] for off in offsets
            ]
            v_offsets_py: list[int] = [vocab_offsets[k] for k in fea_keys]

            gather_meta.append(
                {
                    "dim": dim,
                    "fea_len": fea_len,
                    "fea_keys": list(fea_keys),
                }
            )
            gather_indices.append(
                ConstantBuffer(jnp.array(indices_py, dtype=jnp.int32))
            )
            gather_v_offsets.append(
                ConstantBuffer(jnp.array(v_offsets_py, dtype=jnp.int32))
            )
            self.feature_order_per_dim[dim].extend(fea_keys)

        self.gather_meta = gather_meta
        self.gather_indices = nnx.List(gather_indices)
        self.gather_v_offsets = nnx.List(gather_v_offsets)

    def __call__(self, side_info_flat: jnp.ndarray) -> dict[int, jnp.ndarray]:
        dim_outputs = {}
        for i, meta in enumerate(self.gather_meta):
            dim = meta["dim"]
            fea_len = meta["fea_len"]
            indices = self.gather_indices[i][...]
            v_offsets = self.gather_v_offsets[i][...]

            v_offsets = v_offsets.astype(side_info_flat.dtype)

            ids = jnp.take(side_info_flat, indices, axis=1)
            chex.assert_rank(ids, 3)

            ids_with_offset = ids + v_offsets.reshape(1, -1, 1)

            weight = getattr(self, f"emb_{dim}")
            emb = weight[ids_with_offset]
            chex.assert_shape(emb, (None, len(meta["fea_keys"]), fea_len, dim))

            if fea_len > 1:
                mask = (ids != 0)[..., None]
                emb = jnp.where(mask, emb, 0.0)
                summed = jnp.sum(emb, axis=2)
                if self.pooling == "mean":
                    lengths = jnp.maximum(jnp.sum(mask, axis=2), 1.0)
                    summed = summed / lengths
            else:
                summed = emb[:, :, 0, :]

            if dim not in dim_outputs:
                dim_outputs[dim] = []
            dim_outputs[dim].append(summed)

        out = {}
        for dim, tensor_list in dim_outputs.items():
            out[dim] = jnp.concatenate(tensor_list, axis=1)

        return out

    @staticmethod
    def cast_outputs(
        outputs: dict[int, jnp.ndarray], compute_dtype: jnp.dtype
    ) -> dict[int, jnp.ndarray]:
        """Leave the FP32 pooling island at an explicit compute-dtype boundary."""
        return cast_floating_tree(outputs, compute_dtype)

    @staticmethod
    def infer_compute_dtype(module: nnx.Module) -> jnp.dtype:
        if not isinstance(module, nnx.Module):
            raise TypeError(
                f"module must be flax.nnx.Module, got {type(module).__name__}"
            )
        _, _, ordinary_params, _, _ = nnx.split(
            module, Fp32Param, nnx.Param, ConstantBuffer, ...
        )
        ordinary_param_dtypes = {
            jnp.dtype(leaf.dtype)
            for leaf in jax.tree_util.tree_leaves(ordinary_params)
            if hasattr(leaf, "dtype")
            and jnp.issubdtype(jnp.dtype(leaf.dtype), jnp.floating)
        }
        if len(ordinary_param_dtypes) == 1:
            return next(iter(ordinary_param_dtypes))
        if len(ordinary_param_dtypes) > 1:
            raise TypeError(
                "ordinary floating parameters must share one compute dtype, got "
                f"{sorted(map(str, ordinary_param_dtypes))}"
            )
        if not ordinary_param_dtypes:
            raise TypeError(
                "cannot determine compute dtype: module has no ordinary "
                "floating parameters"
            )


__all__ = ["CategoricalEmbedding"]
