# Flare 与 Fluxway 数据来源适配

本文档说明 Flare 与 Fluxway 之间的公共数据来源适配契约与执行规范。

## 模块职责

| 入口 | 职责 |
| --- | --- |
| `src/flare/data/connectors/adapters/iceberg/source.py` | 原生快照发现、文件区间与消费位置提取 |
| `src/flare/runtime/fluxway/source.py` | 将文件区间与位置转换为调度扫描结果与读取范围 |
| `src/flare/runtime/fluxway/workers/reader.py` | 通用读取进程、任务激活与检查点校验 |
| `src/flare/data/pipeline/consumption/dataloader` | 文件读取、解码、缓冲及设备交付 |

来源边界由扫描规划器和物理读取器两部分组成。`ControlledSourceScanAdapter` 提供
opaque 初始游标、`scan_next(cursor, budget)` 和不访问存储的
`advance_cursor(cursor, records)`。扫描返回有序、类型化的来源单元和游标元数据；
规划器拥有与特定格式有关的版本状态。`SourceReader` 提供分配区间的普通读取与批次
读取。来源工厂负责构造这两类接口。

`build_reader_source()` 将 `WorkItemScheduledDatasetSourceDescriptor` 与列投影、
读取并发和格式选项组合起来，或直接接收 Fluxway 读取会话；这两种模式互斥。
分配模式由编译后的 Fluxway 图推导拓扑并分配工作：

```python
from flare.runtime.fluxway.workers.reader_contracts import build_reader_source
from flare.runtime.fluxway.source import WorkItemScheduledDatasetSourceDescriptor

source = build_reader_source(
    descriptor=WorkItemScheduledDatasetSourceDescriptor(
        source_factory=source_factory,
        global_batch_size=1024,
        completion_policy="pad",
    ),
    projection=("feature", "label", "group_id"),
    source_read_concurrency=4,
    storage_options={
        "storage_format": "parquet",
        "dataset_path": "/data/dataset",
    },
)
```

`source_factory` 必须实现同时创建 scan planner 和 Reader 的 `SourceFactory`。
Iceberg 选项通过 `storage_options` 传入；使用配置文件时不能再同时提供单项目录参数。

`AssignedSourceBatchReader` 将取得的 `SourceWorkItem` 转换为 `SourceRecordRange`。
物化前会核验 step、数据源、schema、revision 和物理 identity。Synthetic padding work
不包含物理文件区间。仅有 padding 的读取需要来源显式提供 Arrow schema，且不会读取数据。
Reader worker 不轮询目录、不分配文件，也不维护独立的持久化来源游标。

## 调度窗口与批次对齐

- **窗口与激活**：Source window 是预取与调度边界，不是数据丢弃边界。窗口结束前已经获取但尚未消费的 `SourceWorkItem` 保留在 canonical pending 状态，并在下一次连续 activation 中继续处理；同一个 `SourceWorkKey` 只能被 lease 和 acknowledge 一次。读取 worker 在尚未发布的 step 异步等待下一次 activation，或接收 `SOURCE_EXHAUSTED`。
- **有效样本计数**：分配式 Reader 对每个批次（包括完整批次）都携带全局有效前缀计数；排名传输将其表示为一个 int32 `samples_valid_count` 标量。终端 partial batch 仅在 `pad` 策略下以固定形状发出；损失、稀疏查表准备及评估指标都会屏蔽计数以外的行。Reader 描述符分别把 `carry`、`drop`、`pad` 和 `require` 映射为保留剩余、丢弃、生成 padding 和拒绝不完整尾部的布局模式。

## 保存与恢复

1. Fluxway source-control 组件保存扫描状态与联合切点。读取 worker 通过 `source_checkpoint_registration()` 注册 `StateCheckpointComponent`，将 Reader position 与切点核对，不保存来源游标、物理文件偏移或预取 payload。
2. 恢复模式采用 `RestoreTopologyMode.GLOBAL_RESHARD`：恢复时首先重建逻辑位置，再根据当前运行时的 worker 拓扑重新分片分配。不保存旧进程的临时迭代器或预取缓冲区。
