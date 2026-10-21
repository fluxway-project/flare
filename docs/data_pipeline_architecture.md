# Data Pipeline Architecture

This document defines the architectural responsibilities and module boundaries of `src/flare/data`.

## Module Boundaries

The public data layer strictly separates production from consumption. Both sides share generic format adapters and structural value objects, but maintain completely decoupled lifecycles, authentication mechanisms, and recovery rules. Consumption serves model training, evaluation, and downstream tasks; production does not depend on model state or consumer progress.

```text
src/flare/data/
├── contracts/       Typed stage inputs, outputs, and structural contracts
├── schema/          Generic structural layouts and batch transforms
├── sources/         Intervals (RecordRange), physical locators, and revision values
├── connectors/      External storage and format adapters (Iceberg, Parquet, etc.)
├── execution/       Production planning and execution assembly
└── pipeline/
    ├── production/  Row parsing, processing, and publishing
    └── consumption/ Range reading, batch assembly, and device delivery
```

| Domain Fact / Responsibility | Owning Component |
| --- | --- |
| External snapshots, files, and native metadata | `connectors/` format adapters |
| File intervals and physical locations | Value objects in `sources/` |
| Config parsing and source object construction | `pipeline/consumption/bootstrap/factory.py` |
| Range to execution work item translation | `src/flare/runtime/fluxway/source.py` |
| Task leases and global consumption progress | Control plane planner, solver, and task ledger |
| File decoding, batch assembly, and device delivery | Flare readers, `schema`, and `dataloader` |
| Joint checkpoint of model and source progress | `checkpoint` contract component registry |

File row numbers, scan positions, and execution steps belong to different coordinate spaces. Flare readers decode and materialize assigned ranges; they do not discover tables or invent global cursors. Model and source states are joined exclusively through the checkpoint contract.
