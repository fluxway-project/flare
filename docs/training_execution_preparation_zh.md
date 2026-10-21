# 训练执行准备

训练 worker 提供框架无关的准备接口
`TrainStepExecutor.prepare_execution(inputs, targets)`。后端可以在首个输入
签名真正执行前，通过该接口准备可执行程序。

准备接口遵守以下契约：

- 可以针对给定的输入签名降低并编译后端 kernel；
- 不得修改模型、优化器、梯度累积或 step 状态；
- 不得执行优化器更新、消费 source work、确认 lease，也不得执行 dummy batch；
- 第一个真实 batch 仍然是传给 `micro_step` 的唯一 payload。

分布式 JAX executor 使用真实的输入和 target 签名准备首个 micro-step、累积
以及 macro-update kernel。它只把抽象的 shape、dtype 和 sharding 交给编译器，
不会在 host 或 device 上物化 batch 大小的副本或零梯度 buffer。准备结果按 key、
shape 和 dtype 缓存，相同签名不会重复编译。没有独立准备工作的后端实现同一个
接口为空操作。Fluxway 只依赖 executor 契约，不了解 JAX、XLA 或任何后端编译机制。

编译失败会传播给所属 operator。worker 不会重试、固定等待、缩小 batch、替换
路径或静默绕过准备阶段。

启用 execution tracing 后，adapter 会记录
`operator_prepare_begin/end/error` 和 `operator_invocation_begin/end/error`，并
带有 step、rank、SPMD 标识以及单调递增的 sequence。这样可以区分首轮编译、
后续执行和协调失败，同时不改变 Fluxway 的调度语义。
