# 流式数据源

Flare 将来源格式发现与按调度读取分开；Fluxway 负责工作分配、执行进度和联合检查点
切点。有限数据集与持续追加来源使用相同的职责边界。

## 架构与数据路径

生产者正常提交存储事务，例如通过 Ray Data 追加 Iceberg 数据。生产者不发布训练游标、
消费任务分配或结束标记。消费者自行选择来源身份、快照模式和尾部策略。

```mermaid
flowchart LR
    A[存储事务] --> B[有界来源扫描]
    B --> C[Fluxway 布局与工作账本]
    C --> D[Reader 实际读取]
    D --> E[下游计算]
    E --> F[联合检查点切点]
```

`SourceFactory.create_scan_planner()` 提供 `ControlledSourceScanAdapter`。
`scan_next(cursor, budget)` 返回类型化的 `SourceScanResult` 元数据，包括有序布局
单元、单元游标、来源与 schema 身份、下一个 opaque cursor 和来源状态。
`advance_cursor()` 在不读取表中数据行的情况下推进原生游标。Iceberg 适配器根据
快照 manifest 中的文件和行数规划区间。

`SourceFactory.create_reader()` 提供物理读取所需的 `SourceReader`。
`AssignedSourceBatchReader` 将真实分配区间解码为 `SourceRecordRange`，核对身份
及边界，并以有界并发读取。Worker 不独立发现文件、轮询目录或推进持久化来源游标。

## 调度与进度

`WorkItemScheduledDatasetSourceDescriptor` 从 Fluxway 编译的执行周期推导 Reader
需求。Fluxway 的布局编译器与逻辑工作账本负责分配、lease 和 acknowledge。
放置与路由使用实际 worker 拓扑；recipe 不重复计算 worker 索引，也不能仅凭
worker 数量推断节点局部性。

Source window 限制任务准入与预取规模，其结束不是数据丢弃边界。尚未消费的逻辑
工作由所属 owner 保留到下一次连续 activation。来源适配器用类型化扫描状态报告
有限来源耗尽，或持续来源正在等待更多数据。

## 批次补齐与终端状态

来源描述将选定策略映射为 Fluxway 尾部布局模式：

| 策略 | 布局模式 | 不完整的剩余数据 |
| --- | --- | --- |
| `carry` | `RETAIN_PENDING` | 保留以供后续工作使用，不作为 partial batch 发出。 |
| `drop` | `DISCARD_REMAINDER` | 丢弃。 |
| `pad` | `SYNTHESIZE_PADDING` | 与类型化 synthetic padding work 合并。 |
| `require` | `REJECT_REMAINDER` | 直接报错。 |

Synthetic padding 不是物理来源区间。仅包含 padding 的 Reader 使用显式 Arrow
schema，不读取数据。分配批次携带全局有效前缀计数；排名任务的传输将其表示为一个
int32 `samples_valid_count` 标量，完整批次也包含该标量。下游损失、稀疏物化和指标
根据该计数屏蔽补齐行。

有限运行保留或丢弃尾部后，可能仍有数据从未进入模型。来源行数、实际发出的有效
样本数和尾部策略必须单独对账，不能由完成步数推断。

## 检查点与恢复

Fluxway 在共同持久化切点协调来源扫描状态、已提交工作进度和下游算子状态。
原生游标 payload 仍由来源适配器拥有。Reader 检查点保存并校验共同 position，
不保存第二份来源游标或旧进程的文件迭代器。

恢复时先建立已提交逻辑位置，再按当前拓扑重新规划未提交工作。Arrow 表、NumPy
缓冲区、对象引用和预取数据都是临时状态。故障后物理读取可能重复；exactly-once
验收针对模型与数据的已提交进度，需要在持久化切点上执行崩溃与恢复对账。
仅有 issued-work 审计或正常退出后的拓扑变更不能证明这一性质。

来源构造与检查点注册方式见[来源适配契约](fluxway_source_adapter_zh.md)。
