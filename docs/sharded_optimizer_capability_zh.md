# 分片优化器架构与能力契约

在 ZeRO-1 和 ZeRO-2 分布式训练中，优化器状态（如 Adam/AdamW 的一阶与二阶动量）占用了大量的加速器显存。Flare 采用分片参数与梯度更新机制，使每个设备仅维护一部分参数及其优化器状态，显著降低显存占用。

Flare 没有重复实现一套分布式优化器，而是直接利用 Flax NNX 标准的单机优化器算子：在更新前将参数和梯度切片到局部分片叶子节点，在各分片上执行标准 NNX 优化器更新，最后通过 `all_gather` 重建前向所需的完整参数（ZeRO-1）。

## 数学原理：切片可交换性

在局部切片上直接执行优化器更新，能够与全量更新后再切片等价的充要条件是：**切片投影算子 $S_r$ 与优化器更新算子 $U$ 满足交换律**：

$$
S_r\left(U(W, G; \theta_{\text{opt}})\right) = U\left(S_r(W), S_r(G); \theta_{\text{opt}}\right), \quad \forall r \in \{0, \dots, K-1\}
$$

- **逐元素更新**：Adam、AdamW、SGD、RMSprop、Adagrad 等优化器每个维度的更新仅依赖该维度的参数、梯度及标量超参数，各坐标相互独立，切片严格可交换。
- **全局归约算子冲突**：若优化器内嵌了全局模长裁剪（如 `optax.clip_by_global_norm`），全局范数依赖全量维度的平方和。在局部切片上计算范数会导致不同设备缩放系数不一致，破坏数值正确性。此类 factory 不获得分片能力凭证；请求分片执行时，静态校验直接抛出 `ValueError`。

## 架构核心组件

1. **`ShardUpdateCapability`**：不可变的能力凭证，仅在优化器满足逐元素独立更新且无内部全局归约时授予。
2. **`ShardedNnxOptimizer`**：继承自 `nnx.Optimizer`，显式绑定 `_flare_shard_update_capability`，遵循开闭原则与里氏替换原则。
3. **`NnxShardUpdatePlan`**：在 JIT 编译前静态遍历模型与优化器状态树，验证 1:1 路径绑定、标量参数拦截与维度对齐。若凭证缺失或路径不匹配则立即报错。
4. **`NnxShardUpdateRuntime`**：在 JIT 内将全量状态安全切片为分片状态，执行 NNX 更新，并通过集合通信 `all_gather` 同步参数。

## 分布式梯度裁剪

优化器内部的裁剪变换无法提供分布式全局范数。分布式裁剪实现必须依据梯度的实际分布进行归约：
1. 各 rank 计算局部分片梯度的平方和：$\text{local\_norm\_sq}_r = \sum_{g \in \text{shard}_r} \|g\|_2^2$；
2. 通过 `all_reduce_sum` 汇总全局标量：$\text{global\_norm} = \sqrt{\sum_r \text{local\_norm\_sq}_r}$；
3. 全局统一缩放梯度后再送入分片优化器更新。

以上公式描述所需的数值契约。能力凭证本身不会安装此归约，优化器的
`grad_clip_norm` 参数也不会启用分布式裁剪。计算范数时，不能把每个 rank 上复制的
同一梯度重复计入。当前分片路径会拒绝带有优化器内部裁剪的 factory。
