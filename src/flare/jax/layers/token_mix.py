from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn import initializers

from flare.jax.layers.base import MLP
from flare.jax.layers.initialization import require_rngs
from flare.jax.layers.numerics import fp32_mean, fp32_rms_normalize, fp32_softmax


class TokenMix(nnx.Module):
    """Token mixing (old_emb style): slice+concat across tokens, LN, FFN, LN, mean pooling."""

    def __init__(
        self,
        token_num: int,
        token_dim: int,
        layer_size: int,
        hidden_dims: tuple[int, ...] = (),
        rngs: nnx.Rngs | None = None,
    ):
        rngs = require_rngs(rngs)
        slice_dim = token_dim // token_num
        ffn_dims = list(hidden_dims) + [token_dim]
        self.layer_size = layer_size
        self.token_num = token_num
        self.token_dim = token_dim
        self.slice_dim = slice_dim
        self.mix_lns = nnx.List()
        self.mix_ffns = nnx.List()
        self.ffn_lns = nnx.List()
        for _ in range(layer_size):
            for _ in range(token_num):
                self.mix_lns.append(nnx.LayerNorm(token_dim, epsilon=1e-3, rngs=rngs))
                self.mix_ffns.append(
                    MLP(token_dim, ffn_dims, activate="gelu", rngs=rngs)
                )
                self.ffn_lns.append(nnx.LayerNorm(token_dim, epsilon=1e-3, rngs=rngs))

    def __call__(self, token_inputs: jnp.ndarray) -> jnp.ndarray:
        token_output = [token_inputs[:, j, :] for j in range(self.token_num)]
        idx = 0
        for _ in range(self.layer_size):
            concat_list = []
            for j in range(self.token_num):
                slices = [
                    t[..., j * self.slice_dim : (j + 1) * self.slice_dim]
                    for t in token_output
                ]
                concat_list.append(jnp.concatenate(slices, axis=-1))
            new_output = []
            for j in range(self.token_num):
                add_token = self.mix_lns[idx](token_output[j] + concat_list[j])
                ffn_out = self.mix_ffns[idx](add_token)
                new_output.append(self.ffn_lns[idx](add_token + ffn_out))
                idx += 1
            token_output = new_output
        return fp32_mean(jnp.stack(token_output, axis=1), axis=1)


class BatchedExpertMoE(nnx.Module):
    """Vectorized mixture-of-experts layer with batched expert parameters."""

    def __init__(
        self,
        batch_dim: int,
        in_dim: int,
        scale_size: int,
        num_experts: int,
        sparse: bool,
        top_k: int,
        rngs: nnx.Rngs,
        residual_norm: bool = True,
    ):
        self.batch_dim = batch_dim
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.sparse = sparse
        self.top_k = min(top_k, num_experts)
        self.residual_norm = residual_norm
        hidden_dim = in_dim * scale_size
        self.hidden_dim = hidden_dim

        init_fn = initializers.lecun_normal()

        if num_experts > 1:
            w_gate = init_fn(rngs(), (batch_dim, in_dim, num_experts))
            self.w_gate = nnx.Param(w_gate)
            self.b_gate = nnx.Param(jnp.zeros((batch_dim, num_experts)))

        w_up = init_fn(rngs(), (batch_dim, num_experts, in_dim, hidden_dim * 2))
        self.w_up = nnx.Param(w_up)
        self.b_up = nnx.Param(jnp.zeros((batch_dim, num_experts, hidden_dim * 2)))

        w_down = init_fn(rngs(), (batch_dim, num_experts, hidden_dim, in_dim))
        self.w_down = nnx.Param(w_down)
        self.b_down = nnx.Param(jnp.zeros((batch_dim, num_experts, in_dim)))
        self.norm_scale = nnx.Param(jnp.ones((batch_dim, in_dim)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.num_experts == 1:
            up_gate_out = jnp.einsum(
                "dbi,dih->dbh", x, self.w_up[...][:, 0]
            ) + jnp.expand_dims(self.b_up[...][:, 0], axis=1)
        else:
            up_gate_out = jnp.einsum(
                "dbi,deih->dbeh", x, self.w_up[...]
            ) + jnp.expand_dims(self.b_up[...], axis=1)
        up, gate_val = jnp.split(up_gate_out, 2, axis=-1)
        hidden = jax.nn.silu(gate_val) * up

        if self.num_experts == 1:
            moe_out = jnp.einsum(
                "dbh,dhi->dbi", hidden, self.w_down[...][:, 0]
            ) + jnp.expand_dims(self.b_down[...][:, 0], axis=1)
        else:
            expert_outs = jnp.einsum(
                "dbeh,dehi->dbei", hidden, self.w_down[...]
            ) + jnp.expand_dims(self.b_down[...], axis=1)
            gate_logits = jnp.einsum(
                "dbi,dio->dbo", x, self.w_gate[...]
            ) + jnp.expand_dims(self.b_gate[...], axis=1)

            if self.sparse:
                top_k_val, top_k_idx = jax.lax.top_k(gate_logits, self.top_k)
                gate_weights = fp32_softmax(top_k_val, axis=-1)
                e_outs_t = jnp.transpose(expert_outs, (0, 1, 3, 2))
                idx_expanded = jnp.expand_dims(top_k_idx, axis=2)
                idx_broadcast = jnp.broadcast_to(
                    idx_expanded, (self.batch_dim, x.shape[1], self.in_dim, self.top_k)
                )
                gathered = jnp.take_along_axis(e_outs_t, idx_broadcast, axis=3)
                moe_out = jnp.einsum("dbik,dbk->dbi", gathered, gate_weights)
            else:
                gate_weights = fp32_softmax(gate_logits, axis=-1)
                moe_out = jnp.einsum("dbe,dbei->dbi", gate_weights, expert_outs)

        if self.residual_norm:
            out = fp32_rms_normalize(x + moe_out, axis=-1, epsilon=1e-5)
            out = out * jnp.expand_dims(self.norm_scale[...], axis=1)
            return out
        return moe_out


class TokenMixLargeMoe(nnx.Module):
    """Multi-head TokenMix + MoE FFN token mixer."""

    def __init__(
        self,
        layer_num: int,
        token_num: int,
        token_dim: int,
        head_num: int = 12,
        scale_size: int = 4,
        num_experts: int = 1,
        sparse: bool = False,
        top_k: int = 2,
        rngs: nnx.Rngs | None = None,
    ):
        rngs = require_rngs(rngs)
        if token_dim % head_num != 0:
            raise ValueError(
                f"token_dim {token_dim} must be divisible by head_num {head_num}"
            )
        self.layer_num = layer_num
        self.token_num = token_num
        self.token_dim = token_dim
        self.head_num = head_num
        self.head_dim = token_dim // head_num
        self.scale_size = scale_size
        self.num_experts = num_experts
        self.sparse = sparse
        self.top_k = min(top_k, num_experts)

        self.head_moe_layers = nnx.List()
        self.revert_moe_layers = nnx.List()

        for _ in range(layer_num):
            self.head_moe_layers.append(
                BatchedExpertMoE(
                    batch_dim=head_num,
                    in_dim=token_num * self.head_dim,
                    scale_size=scale_size,
                    num_experts=num_experts,
                    sparse=sparse,
                    top_k=self.top_k,
                    rngs=rngs,
                )
            )
            self.revert_moe_layers.append(
                BatchedExpertMoE(
                    batch_dim=token_num,
                    in_dim=token_dim,
                    scale_size=scale_size,
                    num_experts=num_experts,
                    sparse=sparse,
                    top_k=self.top_k,
                    rngs=rngs,
                )
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch = x.shape[0]
        for i in range(self.layer_num):
            mix_reshape = x.reshape(batch, self.token_num, self.head_num, self.head_dim)
            mix_transpose = mix_reshape.transpose(2, 0, 1, 3)
            mix_out = mix_transpose.reshape(
                self.head_num, batch, self.token_num * self.head_dim
            )
            head_next_token_stack = self.head_moe_layers[i](mix_out)
            revert_reshape = head_next_token_stack.reshape(
                self.head_num, batch, self.token_num, self.head_dim
            )
            revert_transpose = revert_reshape.transpose(2, 1, 0, 3)
            revert_out = revert_transpose.reshape(
                self.token_num, batch, self.head_num * self.head_dim
            )
            x_t = self.revert_moe_layers[i](revert_out)
            x = x_t.transpose(1, 0, 2)
        return fp32_mean(x, axis=1)


class TokenMixLargePreGateNoPool(nnx.Module):
    """Multi-head TokenMix + MoE FFN token mixer with pre-layer gate3 modulation and no pooling."""

    def __init__(
        self,
        layer_num: int,
        token_num: int,
        token_dim: int,
        head_num: int = 12,
        scale_size: int = 4,
        num_experts: int = 1,
        sparse: bool = False,
        top_k: int = 2,
        residual_enabled: bool = False,
        residual_layers: tuple[int, ...] = (),
        rngs: nnx.Rngs | None = None,
    ):
        rngs = require_rngs(rngs)
        if token_dim % head_num != 0:
            raise ValueError(
                f"token_dim {token_dim} must be divisible by head_num {head_num}"
            )
        self.layer_num = layer_num
        self.token_num = token_num
        self.token_dim = token_dim
        self.head_num = head_num
        self.head_dim = token_dim // head_num
        self.scale_size = scale_size
        self.num_experts = num_experts
        self.sparse = sparse
        self.top_k = min(top_k, num_experts)
        self.residual_enabled = residual_enabled
        self.residual_layers = frozenset(residual_layers)
        self.block_norms = nnx.Dict()
        self.head_norms = nnx.List()
        self.revert_norms = nnx.List()

        self.head_moe_layers = nnx.List()
        self.revert_moe_layers = nnx.List()

        for _ in range(layer_num):
            self.head_moe_layers.append(
                BatchedExpertMoE(
                    batch_dim=head_num,
                    in_dim=token_num * self.head_dim,
                    scale_size=scale_size,
                    num_experts=num_experts,
                    sparse=sparse,
                    top_k=self.top_k,
                    residual_norm=False,
                    rngs=rngs,
                )
            )
            self.revert_moe_layers.append(
                BatchedExpertMoE(
                    batch_dim=token_num,
                    in_dim=token_dim,
                    scale_size=scale_size,
                    num_experts=num_experts,
                    sparse=sparse,
                    top_k=self.top_k,
                    residual_norm=False,
                    rngs=rngs,
                )
            )
            self.head_norms.append(nnx.RMSNorm(token_num * self.head_dim, rngs=rngs))
            self.revert_norms.append(nnx.RMSNorm(token_dim, rngs=rngs))
        for layer_idx in self.residual_layers:
            if not 0 <= layer_idx < layer_num:
                raise ValueError(
                    f"residual layer index {layer_idx} is outside [0, {layer_num})"
                )
            self.block_norms[str(layer_idx)] = nnx.RMSNorm(token_dim, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, pre_layer_gates: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        batch = x.shape[0]
        for i in range(self.layer_num):
            if pre_layer_gates is not None:
                gate = pre_layer_gates[:, i * self.token_dim : (i + 1) * self.token_dim]
                gate = jnp.expand_dims(gate, axis=1)
                x = x * gate

            block_input = x
            if self.residual_enabled and i in self.residual_layers:
                x = self.block_norms[str(i)](x)

            mix_reshape = x.reshape(batch, self.token_num, self.head_num, self.head_dim)
            mix_transpose = mix_reshape.transpose(2, 0, 1, 3)
            mix_out = mix_transpose.reshape(
                self.head_num, batch, self.token_num * self.head_dim
            )
            head_next_token_stack = mix_out + self.head_moe_layers[i](
                self.head_norms[i](mix_out)
            )
            revert_reshape = head_next_token_stack.reshape(
                self.head_num, batch, self.token_num, self.head_dim
            )
            revert_transpose = revert_reshape.transpose(2, 1, 0, 3)
            revert_out = revert_transpose.reshape(
                self.token_num, batch, self.head_num * self.head_dim
            )
            x_t = revert_out + self.revert_moe_layers[i](
                self.revert_norms[i](revert_out)
            )
            x = x_t.transpose(1, 0, 2)
            if self.residual_enabled and i in self.residual_layers:
                x = x + block_input

        return x


__all__ = [
    "BatchedExpertMoE",
    "TokenMix",
    "TokenMixLargeMoe",
    "TokenMixLargePreGateNoPool",
]
