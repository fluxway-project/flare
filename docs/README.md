# Flare-JAX Documentation

Flare-JAX is organized into a public generic framework layer and scenario-specific recipes.
The public layer documents stable contracts for data pipelines, streaming sources, runtime adapters,
training precision, distributed optimizers, training preparation, and inference weights.

## Public Contracts

| Topic | English | 中文 |
| --- | --- | --- |
| Data Architecture | [data_pipeline_architecture.md](data_pipeline_architecture.md) | [data_pipeline_architecture_zh.md](data_pipeline_architecture_zh.md) |
| Dataloader Execution | [dataloader.md](dataloader.md) | [dataloader_zh.md](dataloader_zh.md) |
| Streaming Sources | [streaming_source.md](streaming_source.md) | [streaming_source_zh.md](streaming_source_zh.md) |
| Fluxway Source Adapter | [fluxway_source_adapter.md](fluxway_source_adapter.md) | [fluxway_source_adapter_zh.md](fluxway_source_adapter_zh.md) |
| Data and Checkpoint Validation | [validation.md](validation.md) | [validation_zh.md](validation_zh.md) |
| Training Precision | [jax_training_precision.md](jax_training_precision.md) | [jax_training_precision_zh.md](jax_training_precision_zh.md) |
| Sharded Optimizer Capability | [sharded_optimizer_capability.md](sharded_optimizer_capability.md) | [sharded_optimizer_capability_zh.md](sharded_optimizer_capability_zh.md) |
| Sparse Embedding Architecture | [sparse_embedding_architecture.md](sparse_embedding_architecture.md) | [sparse_embedding_architecture_zh.md](sparse_embedding_architecture_zh.md) |
| Training Execution Preparation | [training_execution_preparation.md](training_execution_preparation.md) | [training_execution_preparation_zh.md](training_execution_preparation_zh.md) |
| Mesh and Inference Weights | [mesh_and_inference.md](mesh_and_inference.md) | [mesh_and_inference_zh.md](mesh_and_inference_zh.md) |

## Scenario Documentation

Scenario implementations, benchmark evidence, failure records, and execution runbooks live under their respective recipe directories. They do not define public Flare APIs. Recipes depend on public contracts, while the public generic layer never depends on recipes.

## Documentation Guidelines

- English documentation files use standard naming (`<topic>.md`).
- Chinese documentation files use the `_zh` suffix (`<topic>_zh.md`).
- Documents reflect the current codebase contracts concisely without describing intermediate historical steps.
- Private endpoints, table names, credentials, and scratch logs remain strictly outside `docs/`.
