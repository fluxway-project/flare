# 数据与 checkpoint 验证

从仓库根目录执行，并使用与应用相同的 Python 环境。工具只读取已保存的工件，
不启动训练、不修改 catalog，也不修复 checkpoint。

## 持久化 source cut

```bash
python -m tools.validation.inspect_checkpoint_source_cut \
  "$CHECKPOINT_DIR" "$CUT" --expected-reader-workers "$READER_WORKERS"
```

检查 checkpoint 事务已经提交、manifest 摘要匹配、framework/SPMD 切点一致，
且每个 Reader partition 的位置都与目标切点一致。这是离线一致性检查；它不能
证明在发布过程中崩溃后如何恢复。

## Dense 模型与优化器状态

```bash
env -u SYSTEM_ENV -u GRAVITINO_TOKEN JAX_PLATFORMS=cpu \
  python -m tools.validation.audit_dense_checkpoint \
  --checkpoint-root "$CHECKPOINT_DIR" --step "$CUT" \
  --expected-optimizer-step "$OPTIMIZER_UPDATES" \
  --state-definition-factory "$STATE_DEFINITION_FACTORY" \
  --factory-kwargs "$STATE_DEFINITION_KWARGS" \
  --spmd-id "$DENSE_SPMD_ID" --component-id "$DENSE_COMPONENT_ID" \
  --scratch "$RUN_DIR/dense-audit-scratch" \
  --output "$RUN_DIR/dense-audit.json"
```

应用提供 `module:factory`，返回 `NnxTrainingStateDefinition`，并以 JSON 对象
传入 factory 参数。工具通过 `CheckpointRepository` 解析已提交 checkpoint，
再由 `OrbaxCheckpointComponent` 将 dense component 恢复到单个 CPU 设备。
它检查期望的优化器更新次数、NNX optimizer step、模型及优化器状态是否有限，
并报告状态指纹及 checkpoint 文件大小/修改时间清单是否保持一致。

Scratch 和输出必须是保护目录以外的新路径。进程要求 `JAX_PLATFORMS=cpu`，
且不接收 `SYSTEM_ENV` 或 `GRAVITINO_TOKEN`。期望更新次数应按每个实际累积
窗口计算，包含 checkpoint 边界刷新出的不完整窗口；数据 microstep 数不一定
等于优化器更新次数。该审计覆盖 dense 状态，sparse 状态与受控崩溃证据仍需
分别验证。

## Native sparse 状态

```bash
env -u SYSTEM_ENV -u GRAVITINO_TOKEN JAX_PLATFORMS=cpu \
  python -m tools.validation.audit_sparse_checkpoint \
  --checkpoint-root "$CHECKPOINT_DIR" --step "$CUT" \
  --expected-optimizer-step "$OPTIMIZER_UPDATES" \
  --store-factory "$SPARSE_STORE_FACTORY" \
  --factory-kwargs "$SPARSE_STORE_KWARGS" \
  --output "$RUN_DIR/sparse-audit.json"
```

应用的 `module:factory` 返回 `SPMD ID -> (component_id, create_optimizer_enabled_store)`
映射。无参 constructor 创建的 store，其 key/value/optimizer dtype、维度、优化器和 score policy 必须
匹配保存时的配置。工具通过 `SparseEmbeddingCheckpointComponent` 将各 component
恢复到单个 CPU partition，再读取 native `export_page_images` 结构化接口。
工具逐个创建、恢复并释放 store，检查逐页形状对齐、每个维度内 key 唯一、导出完整覆盖 store、values/optimizer 状态有限，
以及每个 key 的更新次数不超过提供的优化器计数；输出按 key 排序的状态指纹，
并检查 checkpoint 文件大小/修改时间清单保持一致。

模型相关的 store 构造由 factory 负责。工具不解码 native payload，也不确认或
提交 checkpoint。使用与 dense 审计相同的 CPU 环境，且不传入源凭据。Sparse
状态可恢复且有限，不等于已证明数值参考等价或崩溃 exactly-once。

## Reader 成功读取范围

```bash
python -m tools.validation.reconcile_reader_audit \
  --phase "$RUN_DIR/phase1_reader.json" "$PHASE1_READERS" 0 "$FIRST_CUT" \
  --phase "$RUN_DIR/phase2_reader.json" "$PHASE2_READERS" "$FIRST_CUT" "$FINAL_CUT" \
  --batch-size "$GLOBAL_BATCH_SIZE" --source-rows "$SOURCE_ROWS" \
  --require-exhaustion --output "$RUN_DIR/reader-coverage.json"
```

每个 phase 依次给出 audit 基础路径、worker 数、起始 step 和结束 step；区间为
左闭右开。工具读取已有的 `<stem>_rank_<rank><suffix>` 文件，其内容必须包含
物理 rank 和 Reader 成功执行后的 records。

验证阶段首尾连续、rank/step/范围合法、source/schema 身份一致，并且每步恰好
覆盖一个完整 global batch 的槽位。`--require-exhaustion` 还要求从零覆盖到
`ceil(source_rows / batch_size)`。只有最后一批可以包含超出真实行数的 padding
槽位；输出分别统计真实行数和 padding。

Reader 在正常关闭时写入这个审计，它记录的是成功读取，不是 Input 接收或模型
消费的持久化确认。被杀死的进程可能没有该文件。因此输出明确保留
`crash_exactly_once_proven=false` 和 `input_receipt_proven=false`。

## Reader lane 节点放置

```bash
python -m tools.validation.validate_reader_locality \
  "$RUN_DIR/allocation.json" --minimum-nodes 2 \
  --output "$RUN_DIR/reader-locality.json"
```

输入包含公开 READY allocation report 的放置字段、编译得到的 Reader execution groups
和明确的 Reader/Input SPMD ID。工具核对计划与实际 `attested_node_ids`，要求
Input 跨越指定数量的节点，并验证每条 lane 的所有 Reader 都与对应 Input 位于
同一节点。Execution groups 必须覆盖全部 Reader rank，每条 lane 至少有两个
Reader；不通过 rank 取模推断映射。

该结果证明编译 lane 映射与实际节点放置一致，不代表已经测量网络字节数或物理
多机性能。`tools.validation.local_ray_cluster` 可在单台物理机上创建 GPU 模拟
集群：将可见 GPU UUID 和一份真实 CPU/内存预算分配给多个 Ray 节点，验证 GPU
计算，并在结束时清理自己创建的集群。它要求显式提供非 loopback 网卡地址，
不修改主机路由，也不停止其他集群。

## 端到端证据

应用应分别保存训练、拓扑恢复、尾批有效性、读取覆盖、节点放置、受控崩溃恢复、
独立权重推理与指标比较的结果。崩溃恢复需要先观察到一个明确的持久化切点，
再观察其后的训练进展并中断，通过公开恢复入口加载同一个切点，最终与更新边界
一致的参考运行核对已提交 source/model/optimizer 状态。切点之后尚未提交的读取
可以重放，不能把它们统计成已提交的重复消费。

指标比较应固定不可变 snapshot，使用对齐的完整评估窗口。AUC/GAUC 有限不能
单独证明指标改善或一般统计意义上的收敛。
