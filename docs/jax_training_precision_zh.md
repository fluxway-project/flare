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
- 主参数权重（Master weights）、优化器浮点状态、参数与输入梯度、分布式规约、Loss/Metrics 计算及检查点序列化均保持 FP32；
- 非 Param 状态保持原有声明的 dtype。

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
