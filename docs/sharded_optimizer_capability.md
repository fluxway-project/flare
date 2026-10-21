# Sharded Optimizer Capability and Architecture

In ZeRO-1 and ZeRO-2 distributed training pipelines, optimizer states (such as first- and second-moment accumulators in Adam/AdamW) and parameter updates are partitioned across accelerators to eliminate redundant memory footprint.

Flare avoids maintaining a separate distributed optimizer library or modifying Optax internals. Instead, it leverages canonical Flax NNX optimizer transformations by slicing parameters and gradients into local shard leaves before the update, running standard NNX optimizer steps on each shard, and gathering updated parameters (`all_gather`) where replication is required.

## Mathematical Foundation: Slicing Commutativity

Executing an optimizer step directly on local parameter and gradient slices without cross-device communication is mathematically exact if and only if the update operator commutes with the slicing projection operator:

$$
S_r\left(U(W, G; \theta_{\text{opt}})\right) = U\left(S_r(W), S_r(G); \theta_{\text{opt}}\right), \quad \forall r \in \{0, \dots, K-1\}
$$

- **Elementwise Optimizers**: Algorithms such as Adam, AdamW, SGD, and RMSprop operate coordinate-wise without cross-parameter dependencies. These strictly commute with slicing.
- **Global Reduction Conflict**: Optimizers embedding global reductions (such as `optax.clip_by_global_norm`) break commutativity when evaluated on local shards because local norms differ from the global tensor norm. Such configurations do not receive a shard-update capability. Requesting sharded execution raises `ValueError` during static validation.

## Architectural Components

1. **`ShardUpdateCapability`**: An immutable capability token issued only when the optimizer factory guarantees elementwise independence and absence of local global-norm transforms.
2. **`ShardedNnxOptimizer`**: A typed subclass of `nnx.Optimizer` carrying the `_flare_shard_update_capability` attribute, respecting Liskov Substitution and Open-Closed principles.
3. **`NnxShardUpdatePlan`**: Performs pre-JIT static validation traversing parameter and optimizer state trees to verify 1:1 path alignment, reject scalar leaves, and validate shardability. Incompatible configurations fail fast.
4. **`NnxShardUpdateRuntime`**: Executes inside JIT to slice full states into sharded views, run standard NNX updates, and gather full weights across the shard mesh.

## Distributed Gradient Clipping

An optimizer-local clipping transform cannot establish the required distributed
norm. A distributed clipping implementation must account for the gradient
layout:
1. Each rank computes the sum of squared gradients on its local slice: $\text{local\_norm\_sq}_r = \sum_{g \in \text{shard}_r} \|g\|_2^2$;
2. An `all_reduce_sum` gathers the scalar sum across all ranks to determine $\text{global\_norm}$;
3. Gradients are scaled globally by $\min\left(1, \frac{C}{\text{global\_norm}}\right)$ before passing to the sharded optimizer.

These equations describe the required numerical contract. The capability token
does not install this reduction, and optimizer-local `grad_clip_norm` does not
enable distributed clipping. Replicated gradients must not be counted once per
rank when computing the norm. The current sharded path rejects a factory with
optimizer-local clipping.
