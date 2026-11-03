# JAX 训练性能分析 (Profiling)

本文档介绍 Flare-JAX 中 JAX 分布式训练性能分析（Profiling）的标准工作流、排查要点与验证方法。

## 训练步 Trace 抓取

性能分析应当在完成 Warmup 且计算图已完成编译后进行。吞吐评估窗口与 Profiling 窗口应严格解耦，且在统计前需显式等待设备执行完成：

```python
import jax

# Warmup 触发 XLA 编译与内存初始化
for _ in range(warmup_steps):
    state, outputs = train_step(state, batch)
jax.block_until_ready((state, outputs))

# 配置 Profiler 选项
options = jax.profiler.ProfileOptions()
options.host_tracer_level = 2
options.python_tracer_level = 0

jax.profiler.start_trace(
    "./jax-trace",
    profiler_options=options,
    create_perfetto_trace=True,
)
try:
    for step in range(2):
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            state, outputs = train_step(state, batch)
            jax.block_until_ready((state, outputs))
finally:
    jax.profiler.stop_trace()
```

### 分布式 Trace 规范
- 各 Rank 或参与分析的 Worker 进程需指定独立的 Trace 存储路径（如 `./jax-trace/rank_${RANK}`）。
- 所有进程必须执行完全一致的集合通信序列，避免集合通信乱序或死锁。
- 在 Trace 窗口前后使用进程间 Barrier 进行对齐同步。
- 严禁将 Profiler 的启动与停止耗时纳入步耗时统计。

## Trace 结果解析与汇总

导出的 Perfetto Trace 可通过 Perfetto UI（`ui.perfetto.dev`）加载可视化，或在 TensorBoard XProf 中分析。

对于单 GPU Trace 目录，可使用标准库脚本生成 GPU 忙碌时间与算子耗时分布汇总：

```bash
python tools/validation/summarize_jax_profile.py ./jax-trace \
  --step-name train_step \
  --output ./jax-profile-summary.json
```

分析摘要时的关键指标边界：
- **GPU 忙碌时间（Busy Time）**：通过多流时间区间的并集（Interval Union）计算，防止并发 Stream 重叠统计。
- **算子并发耗时**：各类别算子总时长在不同 CUDA Stream 间可能存在重叠。
- **NCCL 耗时**：集合通信耗时包含节点间等待与同步开销，不完全等同于实际网卡带宽传输时间。
- **信息脱敏**：Trace 文件中包含主机名、代码路径、张量 Shape 与模型结构信息，外发或提交前需完成脱敏审查。

## 显存分配器（Allocator）对比

评估分配器行为时，应在完全一致的模型结构、批大小、梯度累积步数和精度配置下进行基准测试。环境变量需在导入 JAX 之前指定：

```bash
# Platform 分配器（直接通过 cudaMalloc/cudaFree 分配并归还显存给驱动）
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python train.py

# 默认池化分配器（预分配 BFC 内存池复用 buffer）
XLA_PYTHON_CLIENT_ALLOCATOR=default \
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
  XLA_CLIENT_MEM_FRACTION=0.90 \
  python train.py
```

### 分配器特性对照
- **Platform Allocator**：将闲置显存实时退还给驱动，有助于降低显存峰值驻留，但频繁触发的系统调用与显存同步会带来显著开销。
- **Default (BFC) Allocator**：复用内存池以消除运行时分配停顿。在多任务共享环境或混合负载下，合理配置 `XLA_CLIENT_MEM_FRACTION`，预留空间供 NCCL 通信及 CPU-GPU 暂存缓冲区使用。
- **cuda_async**：在特定支持的 CUDA 环境中可用；需经过实际 Benchmark 验证后再评估是否上线。

## 关键路径瓶颈诊断

| 瓶颈现象 | 常见根因 | 排查策略 |
| :--- | :--- | :--- |
| **GPU Stream 出现较长空隙** | Host 发射延迟、输入流水打断、重编译 | 排查数据预取流水线瓶颈、Host 回调阻塞以及 Buffer 捐赠（`donate_argnums`）是否生效；排查动态 Shape 导致的重复编译。 |
| **NCCL 集合通信耗时过长** | 计算负载不均、掉队进程（Straggler）、拓扑不匹配 | 区分梯度规约（`all-reduce`, `reduce-scatter`）与权重搜集（`all-gather`）；检查通信精度（如 BF16 传输）以及网卡/NVLink 拓扑连接状态。 |
| **计算密集算子过慢** | GEMM 瓦片（Tile）配置不佳、内存排布转换、Scatter 散列开销 | 检查慢速 GEMM 算子及 Tensor Core 命中情况；确认是否存在冗余的 Transpose 排布转换或带宽密集型操作。 |
| **显存异常增长 / OOM** | Buffer 捐赠失效、显存碎片化 | 查看编译期 XLA HLO 显存分配分析，确认被捐赠的 Buffer 是否进入 `alias_size_in_bytes`。 |

根据阿姆达尔定律（Amdahl's law）排定优化优先级：
$$\text{Speedup} = \frac{1}{(1 - f) + \frac{f}{s}}$$
其中 $f$ 为待优化部分在关键路径上的时间占比，$s$ 为该部分的加速比。

## ZeRO 通信拓扑微基准测试

Flare-JAX 提供了针对全局 ZeRO 与节点局部 ZeRO 拓扑的专用通信微基准测试脚本：

```bash
# 全局跨节点 ZeRO 拓扑
python tools/validation/benchmark_zero_topology.py \
  --mode global \
  --nodes 2 \
  --gpus-per-node 8 \
  --elements 67108864 \
  --output /tmp/global-zero.json

# 节点内局部 ZeRO + 全局梯度平均
python tools/validation/benchmark_zero_topology.py \
  --mode node_local \
  --nodes 2 \
  --gpus-per-node 8 \
  --elements 67108864 \
  --output /tmp/node-local-zero.json
```

> [!NOTE]
> 评测总元素数量需能够被 GPU 总数整除。通信微基准测试的耗时收益并不直接等同于整网 `train_core` 的最终加速比，必须结合端到端训练吞吐进行联合验证。

