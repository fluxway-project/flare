"""Subprocess fixture for exercising durable Iceberg recovery boundaries."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa

from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
)
from flare.data.connectors.adapters.iceberg.fixed_schema_append import (
    FixedSchemaAppendIcebergDatasink,
)
from flare.data.connectors.adapters.iceberg.recovery import reconcile_iceberg_in_flight
from flare.data.contracts.publication import PublicationProtocol
from flare.data.pipeline.production.builder.resume_state_manager import (
    ResumeStateManager,
)
from flare.data.pipeline.production.publication.manifest import PublicationManifest
from flare.data.pipeline.production.publication.run_lock import publication_run_lock


def example_table() -> pa.Table:
    return pa.table(
        {"id": [1, 2, 3], "_ds": ["2000-01-01"] * 3, "_hour": ["unknown"] * 3}
    )


def main(root: Path, mode: str) -> None:
    state_path = str(root / ("fresh-state.json" if mode == "repeat" else "state.json"))
    with publication_run_lock(state_path):
        if mode == "lock":
            return
        options = json.loads((root / "catalog.json").read_text())
        provider = ConfiguredIcebergCatalogProvider(options)
        manifest = PublicationManifest.from_files(["source-a", "source-b"])
        state = ResumeStateManager(state_path)
        _, in_flight = state.load()
        reconcile_iceberg_in_flight(
            state, in_flight, catalog_provider=provider, table_identifier="test.events"
        )
        if not any(state.is_pending(path) for path in manifest.files):
            return
        state.mark_in_flight(
            manifest.identity.value,
            list(manifest.files),
            PublicationProtocol.ICEBERG_SNAPSHOT,
        )
        data = example_table()
        with provider.open() as catalog:
            table = catalog.load_table("test.events")
            sink = FixedSchemaAppendIcebergDatasink(
                expected_schema=data.schema,
                expected_iceberg_schema=table.schema(),
                catalog_provider=provider,
                table_identifier="test.events",
                snapshot_properties={"flare.publication-id": manifest.identity.value},
                coalesce_target_bytes=1024 * 1024,
            )
        sink.on_write_start(data.schema)
        result = sink.write([data], None)
        if mode == "before_commit":
            os._exit(85)
        sink.on_write_complete(SimpleNamespace(write_returns=[result]))
        if mode == "after_commit":
            os._exit(86)
        state.complete_in_flight(manifest.identity.value)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
