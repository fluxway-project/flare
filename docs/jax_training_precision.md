# NNX Training Precision

This document specifies the training precision policies, parameter role contracts, and numerical stability invariants for Flax NNX models in Flare-JAX.

## Precision Policies

`NnxTrainingStateDefinition` governs the precision policy for Flax NNX models. The default policy is FP32, with explicit opt-in for BF16 and FP16:

```python
from flare.jax.layers import NnxModelDefinition
from flare.jax.training import NnxTrainingStateDefinition, PrecisionPolicy

model_definition = NnxModelDefinition.from_factory(model_builder)
state_definition = NnxTrainingStateDefinition.from_model_definition(
    model_definition,
    model_builder.create_optimizer,
    precision_policy=PrecisionPolicy.bf16(),
)
```

Under mixed-precision policies:
- Standard `nnx.Param` parameters cast to temporary BF16/FP16 compute views for forward/backward passes;
- Master weights, optimizer floating states, gradients, distributed reductions, loss/metric computations, and checkpoint weights remain FP32;
- Non-Param states retain their declared dtypes.

## Parameter Roles and Embedding Boundaries

`CategoricalEmbedding` uses `Fp32Param` by default, keeping dense embedding tables and pooling operations in FP32. Pooled representations must be explicitly cast to the model compute dtype before concatenating into ordinary dense layers:

```python
from flare.jax.training import Fp32Param

self.embedding_table = Fp32Param(initial_values)
```

Dynamic sparse embeddings support FP16 storage, lookup transport, and pooling under BF16/FP16 policies; their optimizer states and scatter update gradients remain FP32. Non-finite values or incompatible parameter roles raise exceptions immediately without silent truncation or fallback.

## Numerically Sensitive Reductions

Numerically sensitive gating and reduction paths use dedicated helpers:
- `flare.jax.layers.fp32_softmax`
- `flare.jax.layers.fp32_mean`
- `flare.jax.layers.fp32_rms_normalize`

These functions evaluate the reduction in FP32 and cast the output back to the input floating dtype at the boundary, ensuring numerical stability while maintaining compute throughput.
