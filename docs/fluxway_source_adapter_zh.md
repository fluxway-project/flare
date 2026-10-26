# Flare 与 Fluxway 数据来源适配

本文档说明 Flare 与 Fluxway 之间的公共数据来源适配契约与执行规范。

## 模块职责

| 入口 | 职责 |
| --- | --- |
| `src/flare/data/connectors/adapters/iceberg/source.py` | 原生快照发现、文件区间与消费位置提取 |
| `src/flare/data/connectors/adapters/iceberg/scan.py` | 将原生文件区间与游标转换为类型化扫描结果 |
| `src/flare/runtime/fluxway/source.py` | 来源绑定、读取拓扑及会话创建 |
| `src/flare/data/pipeline/consumption/source/execution/record_ranges.py` | 标准文件范围校验与物化 |
| `src/flare/runtime/fluxway/workers/reader.py` | 通用读取进程、任务激活与检查点校验 |
| `src/flare/data/connectors/adapters/parquet/range_reader.py` | 按列与行范围读取、文件缓存及关闭 |
| `src/flare/data/pipeline/consumption/dataloader` | 独立文件加载器、有界读取与缓冲工具 |

来源边界由扫描规划器和物理读取器两部分组成。`ControlledSourceScanAdapter` 提供
不透明初始游标、`scan_next(cursor, budget)` 和不访问存储的
`advance_cursor(cursor, records)`。扫描返回有序、类型化的来源单元和游标元数据；
规划器拥有与特定格式有关的版本状态。`SourceReadSession.read(step, work_items=...)`
返回 `SourceBatch(payload, cardinality, inputs)`，每种来源解释自己的不透明任务，
payload 保留原生表示。来源工厂负责创建接口及校验自己的配置；两个端口都必须提供
`close()`，允许同步或异步实现。装配和物理读取时序见
[数据加载流程](dataloader_zh.md)；追加扫描与恢复时序见[流式数据源](streaming_source_zh.md)。

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

`source_factory` 必须实现 `SourceFactory.prepare_options()`、扫描规划器和会话创建。
公共 builder 不强制列投影或读取并发，各工厂校验自身必需配置。
Iceberg 选项通过 `storage_options` 传入；使用配置文件时不能再同时提供单独的目录服务参数。

`RecordRangeReadSession` 将取得的 `SourceWorkItem` 转换为 `SourceRecordRange`。
物化前会核验执行步、数据源、结构摘要、版本和物理身份。补齐任务不包含物理文件区间。
仅有补齐行的读取需要来源显式提供 Arrow schema，且不会读取数据文件。
读取进程不独立发现新任务、分配文件，也不维护第二份持久化来源游标。

原生查询或分片会话直接实现 `SourceReadSession`，自行解释任务 token 并调用 SDK，
无需经过文件区间解码。默认训练 worker 接受 Arrow Table、RecordBatch 或列映射；
其他表示由实际消费者的 `transform_raw_batch()` 实现处理。接口不能补出后端本身
缺少的可重放游标或并行分片能力。

Iceberg 通过 `IcebergCatalogProvider` 注入目录创建；`IcebergTableSession` 统一
拥有目录上下文、表 UUID 与刷新校验，格式读取器复用表已授权的 FileIO。文件访问、
数据发布和外部血缘独立组合。可选 `LineageConfig` 创建一个 `LineageRecorder`，
只聚合有界元数据并批量交给 `LineageExporter`。输入事实在读取成功后记录，输出事实
在发布确认后记录。描述符将 `include_lineage=self.lineage is not None` 传给
`SourceFactory.create_reader()`，未配置时标准会话不构建输入资产引用，
不创建 exporter 或后台任务。血缘投递不提交数据，
也不拥有来源检查点状态。

`LineageRecorder(config)` 只装配配置，首次非空 flush 才在记录器的操作 owner 内
创建 exporter。调用方取消等待不会丢弃创建、读取或投递；close 停止准入、等待已接收
操作，再释放资源。被等待的操作及其子任务反向调用同一 close 会立即抛错。网络超时
由各后端实现；调用者的关闭等待截止时间不能终止已经运行的同步 SDK 调用。

## 调度窗口与批次对齐

- **窗口与激活**：调度窗口是预取与任务准入边界，不是数据丢弃边界。已获取但尚未消费的 `SourceWorkItem` 由负责组件保留，在下一次连续激活中继续处理；账本统一管理任务租约和确认。读取进程异步等待未发布执行步的激活，或接收 `SOURCE_EXHAUSTED`。
- **有效样本计数**：每个分配式读取批次都携带全局有效前缀计数，完整批次也包含该计数；排名传输将其表示为一个 int32 `samples_valid_count` 标量。终端不完整批次仅在 `pad` 策略下以固定形状发出；损失、稀疏查表准备及评估指标屏蔽计数以外的行。`carry`、`drop`、`pad` 和 `require` 的终端行为见流式文档。

## 保存与恢复

1. Fluxway 来源控制组件保存身份、消费游标与联合切点；消费游标按切点推导，可以落后于扫描位置。读取进程通过 `source_checkpoint_registration()` 注册 `StateCheckpointComponent`，将 `position` 与切点核对，不保存第二份来源游标、物理文件偏移或预取数据。
2. 恢复模式采用 `RestoreTopologyMode.GLOBAL_RESHARD`：恢复时首先重建逻辑位置，再根据当前运行时的工作进程拓扑重新分片分配。不保存旧进程的临时迭代器或预取缓冲区。
