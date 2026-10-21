# 稀疏 Embedding 流水线

Flare 将稀疏特征处理、分区 Embedding 存储和稠密计算组织为 Fluxway 算子。
算子依赖、有界并发和 transport lease 决定哪些工作能够重叠执行。异步 API
保留对应边界的并发能力；整体吞吐和同步开销仍需通过实际运行测量。

## 前向与反向路径

```mermaid
flowchart LR
    R[Reader 读取区间] --> I[Input 去重 ID 并保留还原索引]
    I --> E[分区 Embedding 查表]
    E --> C[Compute 本地 Gather 与前向]
    I --> C
    C --> D[稠密优化器更新]
    C --> G[稀疏梯度路由]
    G --> S[稀疏累积与优化器更新]
```

Input worker 对各稀疏特征组去重，保留还原索引和样本来源信息，并发准备互不
依赖的特征组。Embedding worker 接收路由后的唯一 ID，调用 native store 的
异步查表接口。稀疏 `shard` mesh 与 `PartitionSpec` 决定分区归属；稠密 `dp`
mesh 不定义稀疏分区身份。

Compute worker 接收紧凑 Embedding 值和还原索引，在本地 Gather 为模型所需
的特征布局。CPU 稀疏存储与 CUDA 计算组合时，向量跨越 host/device 边界。
去重减少重复向量传输，实际带宽取决于数据、路由和运行环境。图还需要传递
标签、side information 和有效样本数。

训练将梯度路由回稀疏 owner，并按所选策略更新稠密状态。分布式 Zero-1 训练
包含该策略要求的 collective。独立 Eval 在各 compute worker 加载 replicated
模型权重并执行本地前向。具体契约见 [mesh 与推理权重](mesh_and_inference_zh.md)。

## 特征与存储 schema

应用负责特征命名、特征组解释和模型输入顺序。native store 负责支持的值维度
与分区 schema。Flare 在组合这些 owner 时校验显式映射，不推断应用特征
schema。为了满足存储布局而重排模型特征，会改变模型语义。

## 状态、检查点与推理

sparse_embedding 拥有 native table、优化器状态、checkpoint fragment、lineage
和分区恢复。Fluxway 协调图生命周期与持久化的来源/模型 cut。Flare 组合有类型
的分区映射，并通过该边界传递 owner 的 opaque restore payload。

当 checkpoint cut 位于累积窗口中间，已注册的 finalizer 在 storage staging
前，按实际窗口长度应用 pending dense/sparse update，不额外消费来源批次。
稀疏拓扑变化后，owner 先建立新的 FULL root，再进行后续 incremental save；
发布和 acknowledge 遵循 owner 生命周期。正常退出后的恢复不能代替进程崩溃
情况下的正确性验证。

独立推理只接收模型定义、不可变的 `steps/<step>` 已发布权重和自己的拓扑。
权重目录暴露 `model/` 与 `sparse/`；对应 adapter 加载 model-only 状态和稀疏
推理权重，不读取训练 refs、optimizer、source cursor 或 controller metadata。
缺失权重或不兼容 schema 直接报错。

来源调度、padding 有效性和 Reader 联合位置见 [来源适配器](fluxway_source_adapter_zh.md)。
应用 schema、编译前向、指标和可运行命令见
[recipe 使用说明](../recipes/README.md)。
