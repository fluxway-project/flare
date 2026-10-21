# Flare: Heterogeneous MPMD Machine Learning Framework

Flare is a high-performance, heterogeneous machine learning framework designed for large-scale recommendation systems and deep learning workloads. Built on top of [JAX](https://github.com/google/jax) and [Ray](https://github.com/ray-project/ray), Flare decouples sparse parameter management from dense compute topologies using a Multiple-Program Multiple-Data (MPMD) execution model.

## Key Features

- **Heterogeneous MPMD Pipeline**: Efficiently orchestrates distinct roles (Reader, Input SPMD, Sparse Embedding Workers, Dense Compute Workers, Metric Aggregators) across heterogeneous compute clusters.
- **Sparse-Dense Decoupling**: Offloads terabyte-scale sparse embedding lookup and updates to CPU/memory-optimized nodes while keeping GPU workers saturated with high-throughput dense compute.
- **Zero-Copy & Native Acceleration**: Leverages Cython, Numba, and CuPy extensions for microsecond-level sparse indexing, pooling, and data ingestion.
- **Distributed Precision Management**: First-class support for mixed precision (`float32`, `bfloat16`, `float16`) with targeted numerical rescue mechanisms for sparse parameter accumulation.
- **Pluggable Data Connectors**: Production-grade integration with Apache Iceberg catalogs, S3, and Parquet data lakes.

## Architecture

For an in-depth architectural breakdown, please refer to the documents in `docs/`:

- [Sparse Embedding Architecture](docs/sparse_embedding_architecture.md) ([中文](docs/sparse_embedding_architecture_zh.md))
- [Fluxway Source Adapter](docs/fluxway_source_adapter.md) ([中文](docs/fluxway_source_adapter_zh.md))
- [Sharded Optimizer Capabilities](docs/sharded_optimizer_capability.md) ([中文](docs/sharded_optimizer_capability_zh.md))
- [Training Execution Preparation](docs/training_execution_preparation.md) ([中文](docs/training_execution_preparation_zh.md))
- [Mesh and Inference Weights](docs/mesh_and_inference.md) ([中文](docs/mesh_and_inference_zh.md))

## Installation

### Prerequisites

- Linux (x86_64 or aarch64)
- Python >= 3.10
- CUDA 12+ (for GPU acceleration)

### Building from Source

```bash
git clone https://github.com/fluxway-project/flare.git
cd flare

# Install dependencies and build native extensions
pip install -e ".[dev]"
```

## Running Tests

Flare uses `pytest` for unit and component verification:

```bash
pytest
```

To format and lint code according to the repository standard:

```bash
./tools/format.sh
```

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
