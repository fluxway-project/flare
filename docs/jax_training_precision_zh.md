# NNX 训练精度配置

本文档规范 Flax NNX 模型的训练精度策略、参数角色约定及数值稳定性边界。

## 精度策略

`NnxTrainingStateDefinition` 统管自定义 NNX 模型的精度策略。默认为 FP32，支持显式启用 BF16 或 FP16：

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

启用半精度策略时：
- 普通 `nnx.Param` 参数在模型前向与反向计算中使用临时的 BF16/FP16 计算视图；
- 主参数权重（Master weights）、优化器浮点状态、梯度累积、Loss/Metrics 计算及检查点序列化均保持 FP32；
- BF16/FP16 训练的稠密与稀疏梯度统一使用 BF16 传输，FP32 训练仍使用 FP32 传输；
- 非 Param 状态保持原有声明的 dtype。

梯度传输精度由精度策略直接决定，无需额外开关。BF16 保留 FP32 的指数范围，
避免 FP16 计算模式引入 FP16 传输溢出；量化和集合求和仍存在舍入误差。
规约结果恢复 FP32 后再执行平均和优化器更新。默认边界同步模式中，
ZeRO-1/2/3 先在本地以 FP32 累积完整的 micro-step 梯度，仅在 macro update
边界转换并同步一次；ZeRO-2/3 的全局分片范围使用 reduce-scatter，节点内
分片范围使用一次全局 all-reduce 后在本地切片，得到分片梯度。
显式低显存 micro 模式则每个微步转换并规约一次。两种模式在编译后的 kernel
内复用同一份集合通信实现。参数 all-gather 仍传输 FP32 主权重。

## 参数角色与 Embedding 边界

`CategoricalEmbedding` 默认使用 `Fp32Param` 角色，其密集 embedding 查表与 pooling 计算保持在 FP32。其输出在拼接进入常规 Dense 层前，必须显式转换为模型的 compute dtype。

```python
from flare.jax.training import Fp32Param

self.embedding_table = Fp32Param(initial_values)
```

动态稀疏 Embedding 支持在 BF16/FP16 策略下采用 FP16 进行存储、查表传输和 pooling；其优化器状态与原生 scatter 更新梯度保持 FP32。若声明了不合法的参数类型或发生非有限值计算，系统将直接抛出异常，严禁隐式类型推断、静默截断或 fallback 行为。

## 数值敏感算子

数值敏感的归约与门控路径使用专门的辅助算子：
- `flare.jax.layers.fp32_softmax`
- `flare.jax.layers.fp32_mean`
- `flare.jax.layers.fp32_rms_normalize`

这些算子在内部提升至 FP32 进行高精度数值计算，并在边界处将输出恢复为输入张量的原始浮点类型，避免下游精度污染。
