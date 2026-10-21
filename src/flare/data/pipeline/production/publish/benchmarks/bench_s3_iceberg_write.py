import os
import time

import pyarrow as pa
import ray

from flare.data.connectors.adapters.iceberg.catalog import build_iceberg_catalog_kwargs
from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher
from flare.data.contracts.sink import PublicationIdentity
from flare.data.pipeline.production.publish.resource_profile import (
    DEFAULT_WRITE_TASK_NUM_CPUS,
)


def create_dummy_batch(batch_idx):
    # Create 1000000 rows per batch
    num_rows = 1000000
    return {
        "id": list(range(num_rows)),
        "_ds": ["2025-12-03"] * num_rows,
        "_hour": ["12"] * num_rows,
        "feature_1": [1.0] * num_rows,
        "feature_2": [2.0] * num_rows,
    }


def main():
    ray.init()

    # 32 batches * 1000000 = 32,000,000 rows
    ds = ray.data.range(32, override_num_blocks=32).map_batches(
        create_dummy_batch, batch_format="pyarrow", batch_size=1
    )

    # Repartition equivalent to test large blocks (optional, let's use map_batches)
    # ds = ds.map_batches(lambda x: x, batch_size=1048576, batch_format="pyarrow")

    table_identifier = "bench.iceberg_test_e2e_mock"
    catalog_kwargs = build_iceberg_catalog_kwargs(
        name="rest",
        catalog_type="rest",
        uri=os.environ.get("ICEBERG_REST_URI"),
        warehouse=os.environ.get("ICEBERG_WAREHOUSE"),
    )

    print("Starting write...")
    start = time.time()

    consumer = IcebergSnapshotPublisher(
        table_identifier=table_identifier,
        catalog_kwargs=catalog_kwargs,
        write_task_num_cpus=DEFAULT_WRITE_TASK_NUM_CPUS,
    )
    result = consumer.publish(
        ds,
        schema=pa.schema(
            [
                ("id", pa.int64()),
                ("_ds", pa.string()),
                ("_hour", pa.string()),
                ("feature_1", pa.float64()),
                ("feature_2", pa.float64()),
            ]
        ),
        publication=PublicationIdentity("bench-s3-iceberg-write-v1"),
    )
    total_rows = result.total_rows

    elapsed = time.time() - start
    print(f"Total rows: {total_rows}")
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Throughput: {total_rows / elapsed:.2f} rows/s")


if __name__ == "__main__":
    main()
