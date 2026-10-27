"""Iceberg transactional sink for the Ray Data pipeline."""

from __future__ import annotations

import logging
import time
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import ray

from flare.data.connectors.adapters.iceberg.access import (
    IcebergCatalogProvider,
    IcebergTableSession,
)
from flare.data.connectors.adapters.iceberg.fixed_schema_append import (
    FixedSchemaAppendIcebergDatasink,
)
from flare.data.connectors.adapters.iceberg.observation import (
    IcebergCommitObservation,
    observe_committed_publication,
)
from flare.data.connectors.adapters.iceberg.table import (
    DEFAULT_WRITER_TARGET_SIZE_BYTES,
    validate_iceberg_table,
    validate_iceberg_table_properties,
)
from flare.data.contracts.assets import AssetRef
from flare.data.contracts.sink import PublicationIdentity, SinkResult
from flare.data.pipeline.production.publication.contract import (
    PUBLICATION_ID_PROPERTY,
    PublicationProtocol,
    validate_publication_id,
)

if TYPE_CHECKING:
    import ray.data

logger = logging.getLogger(__name__)

_INFLIGHT_FORMATS: tuple[str, ...] = ("arrow", "numpy")


@dataclass(frozen=True, slots=True)
class IcebergPublication:
    publisher: IcebergSnapshotPublisher
    identity: PublicationIdentity

    def finalize(
        self,
        dataset: ray.data.Dataset,
        *,
        schema: pa.Schema,
    ) -> SinkResult:
        return self.publisher.publish(
            dataset,
            schema=schema,
            publication=self.identity,
        )


class IcebergSnapshotPublisher:
    @property
    def publication_protocol(self) -> PublicationProtocol:
        return PublicationProtocol.ICEBERG_SNAPSHOT

    def __init__(
        self,
        table_identifier: str,
        catalog_provider: IcebergCatalogProvider,
        write_task_num_cpus: float,
        snapshot_properties: dict[str, str] | None = None,
        write_concurrency: int | None = None,
        writer_target_size_bytes: int = DEFAULT_WRITER_TARGET_SIZE_BYTES,
        observe_file_layout: bool = False,
    ):
        self._table_identifier = table_identifier
        if write_task_num_cpus <= 0:
            raise ValueError(
                f"write_task_num_cpus must be positive, got {write_task_num_cpus}"
            )
        self._write_task_num_cpus = write_task_num_cpus
        if writer_target_size_bytes <= 0:
            raise ValueError(
                "writer_target_size_bytes must be positive, "
                f"got {writer_target_size_bytes}"
            )
        self._writer_target_size_bytes = writer_target_size_bytes
        if not isinstance(observe_file_layout, bool):
            raise TypeError("observe_file_layout must be bool")
        self._observe_file_layout = observe_file_layout

        self._catalog_provider = catalog_provider
        self._asset_namespace = catalog_provider.identity
        self._snapshot_properties = dict(snapshot_properties or {})
        if PUBLICATION_ID_PROPERTY in self._snapshot_properties:
            raise ValueError(
                f"snapshot_properties must not define {PUBLICATION_ID_PROPERTY}"
            )
        self._write_concurrency = write_concurrency
        self._load_table_properties()

    def _load_table_properties(self) -> dict[str, str]:
        with closing(
            IcebergTableSession(self._catalog_provider, self._table_identifier)
        ) as session:
            table_properties = dict(session.table().properties)
        validate_iceberg_table_properties(
            table_properties,
            expected_writer_target_size_bytes=self._writer_target_size_bytes,
        )
        return table_properties

    def _observe_committed_publication(
        self,
        *,
        publication_id: str,
        write_seconds: float,
    ) -> IcebergCommitObservation:
        return observe_committed_publication(
            catalog_provider=self._catalog_provider,
            table_identifier=self._table_identifier,
            expected_publication_id=publication_id,
            write_seconds=write_seconds,
            include_file_layout=self._observe_file_layout,
        )

    def bind(self, publication: PublicationIdentity) -> IcebergPublication:
        return IcebergPublication(self, publication)

    def publish(
        self,
        ds: ray.data.Dataset,
        *,
        schema: pa.Schema,
        publication: PublicationIdentity,
    ) -> SinkResult:
        with closing(
            IcebergTableSession(self._catalog_provider, self._table_identifier)
        ) as session:
            expected_iceberg_schema = validate_iceberg_table(
                session.table(),
                expected_schema=schema,
                expected_writer_target_size_bytes=self._writer_target_size_bytes,
            )
        snapshot_properties = dict(self._snapshot_properties)
        publication_id = validate_publication_id(publication.value)
        snapshot_properties[PUBLICATION_ID_PROPERTY] = publication_id
        datasink = FixedSchemaAppendIcebergDatasink(
            expected_schema=schema,
            expected_iceberg_schema=expected_iceberg_schema,
            table_identifier=self._table_identifier,
            catalog_provider=self._catalog_provider,
            snapshot_properties=snapshot_properties,
        )
        write_started = time.perf_counter()
        ds.write_datasink(
            datasink,
            ray_remote_args={"num_cpus": self._write_task_num_cpus},
            concurrency=self._write_concurrency,
        )
        observation = self._observe_committed_publication(
            publication_id=publication_id,
            write_seconds=time.perf_counter() - write_started,
        )
        self._log_commit_observation(observation)
        return SinkResult(
            total_rows=observation.added_records,
            outputs=(
                AssetRef(
                    self._asset_namespace,
                    self._table_identifier,
                    str(observation.snapshot_id),
                ),
            ),
        )

    def _log_commit_observation(self, observation: IcebergCommitObservation) -> None:
        layout = observation.file_layout
        logger.info(
            "Iceberg publication committed: publication=%s snapshot=%d rows=%d "
            "write=%.3fs observation=%.3fs files=%s file_bytes=%s "
            "smallest_file_bytes=%s p50_file_bytes=%s p95_file_bytes=%s "
            "largest_file_bytes=%s",
            observation.publication_id,
            observation.snapshot_id,
            observation.added_records,
            observation.write_seconds,
            observation.snapshot_observation_seconds,
            None if layout is None else layout.data_file_count,
            None if layout is None else layout.data_file_bytes,
            None if layout is None else layout.smallest_file_bytes,
            None if layout is None else layout.p50_file_bytes,
            None if layout is None else layout.p95_file_bytes,
            None if layout is None else layout.largest_file_bytes,
        )
