"""flare.data.pipeline.production.publish — Ray Data streaming data pipeline.

Stage-split pipeline architecture:
  HdfsZstdDatasource          (IO + zstd decompress, 1 ReadTask / file)
    → map_batches(ParseUDF)   [CPU pool, sized by StageTopologyPlanner.n_parse]
    → map_batches(EncodeUDF)  [GPU or CPU pool, ExprPlan-based encoding]
  → DatasetTerminal           (bound publication | stream consumer)

Decoupling parse from the Datasource lets Ray Data scale CPU parsing
independently of file count.
"""
