# Mesh 与推理权重

本文档说明 Fluxway 与 JAX 设备 Mesh 之间的映射关系，以及推理权重的独立恢复契约。

## 图拓扑与 Mesh 映射

密集数据并行（Dense DP）与稀疏表归属（Sparse table ownership）在逻辑上相互解耦。密集 `dp` mesh 不直接校验稀疏表的 `shard` 映射。在图构建阶段，依据请求的稀疏拓扑生成映射，并结合稀疏 worker mesh 校验：

```python
from flare.mesh import (
    bind_sparse_worker_meshes,
    build_1d_device_mesh,
    build_sparse_partition_mappings,
)

dense_mesh = build_1d_device_mesh(1, axis_name="dp")
sparse_workers = {"users": 2, "items": 1}
sparse_mappings = build_sparse_partition_mappings(sparse_workers)
sparse_meshes = bind_sparse_worker_meshes(
    partition_mappings=sparse_mappings,
    worker_counts=sparse_workers,
)
```

稀疏 worker 数量可以与密集 DP 数量不同。非法拓扑尺寸或缺失的分组将在启动前直接报错。

## JAX 设备环境

独立本地推理在每个 worker 内部使用单设备计算 mesh：

```python
import jax
from flare.jax.mesh import ParallelMesh

local_mesh = ParallelMesh.from_dp(
    size=1,
    devices=(jax.local_devices()[0],),
    axis_name="dp",
).mesh
```

分布式训练需显式初始化 JAX 进程拓扑并传入对应物理设备，`materialize_jax_mesh` 针对多 rank 场景要求显式设备映射。

## 独立权重加载

推理服务只需接收模型定义、不可变权重目录与自身拓扑。已发布的权重资产包含且仅包含 `model/` 与 `sparse/` 子目录：模型适配器恢复纯模型 Orbax 状态，稀疏适配器根据分区选项恢复 Lance 权重。推理过程不读取训练检查点引用、优化器状态或控制器元数据。

```python
from sparse_embedding.checkpoint import PartitionSpec
from flare.inference.weights import bind_inference_weights
from flare.jax.layers import NnxModelDefinition

model_definition = NnxModelDefinition.from_factory(
    model_builder,
    sparse_groups=tuple(sparse_workers),
)
weights = bind_inference_weights(
    weight_path,
    model_definition,
    dense_mesh,
    partition_mappings=sparse_mappings,
    sparse_partition_assignment_spec=PartitionSpec.modulo(),
    key_dtype="int64",
)
model_state = weights.model.load(model_definition, local_mesh)
model = model_definition.materialize(model_state)
```

`weight_path` 指向不可变的 `steps/<step>` 目录。映射中的 group 命名必须与模型的稀疏组精确对齐。缺失权重、不兼容 schema 或非法映射均会触发即刻失败。
