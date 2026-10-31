"""Shared pipeline contracts, independent of recipe configurations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
)
from flare.data.contracts.publication import PublicationIdentity
from flare.data.execution.plan.terminal import DatasetTerminal


class TestProtocolConformance:
    """Verify concrete consumers conform to the dataset end-of-stream contract."""

    def test_non_conforming_class_rejected(self):
        class Empty:
            pass

        assert not hasattr(Empty(), "finalize")

    def test_consumer_implementations_satisfy_protocol(self):
        """Bound Iceberg and stream terminals must conform to DatasetTerminal."""

        from unittest.mock import MagicMock, patch

        from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher
        from flare.data.connectors.adapters.iceberg.table import (
            DEFAULT_WRITER_TARGET_SIZE_BYTES,
            iceberg_table_properties,
        )
        from flare.data.connectors.adapters.stream.sink import RayDataStreamDrainer

        table = MagicMock()
        table.properties = iceberg_table_properties(DEFAULT_WRITER_TARGET_SIZE_BYTES)
        inflight_arrow = RayDataStreamDrainer(output_format="arrow")
        inflight_numpy = RayDataStreamDrainer(output_format="numpy")
        with patch(
            "flare.data.connectors.adapters.iceberg.sink.IcebergTableSession",
            return_value=SimpleNamespace(table=lambda: table, close=lambda: None),
        ):
            iceberg_consumer = IcebergSnapshotPublisher(
                table_identifier="db.tbl",
                catalog_provider=ConfiguredIcebergCatalogProvider(
                    {
                        "name": "rest",
                        "type": "rest",
                        "uri": "http://iceberg-rest.local",
                    }
                ),
                write_task_num_cpus=1.0,
            )

        from flare.data.contracts.publication import PublicationIdentity

        def require_terminal(terminal: DatasetTerminal) -> DatasetTerminal:
            return terminal

        assert require_terminal(inflight_arrow) is inflight_arrow
        assert require_terminal(inflight_numpy) is inflight_numpy
        assert require_terminal(iceberg_consumer.bind(PublicationIdentity("abc")))


class TestDatasetTerminal:
    """Behaviour of the stock DatasetTerminal implementations and the
    OCP injection contract on ``RayDataPipeline.execute``."""

    def test_inflight_consumer_rejects_invalid_format(self):
        from flare.data.connectors.adapters.stream.sink import RayDataStreamDrainer

        with pytest.raises(ValueError):
            RayDataStreamDrainer(output_format="parquet")  # type: ignore[arg-type]

    def test_iceberg_consumer_writes_without_pre_repartition(self):
        from unittest.mock import MagicMock, patch

        import pyarrow as pa

        from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher
        from flare.data.connectors.adapters.iceberg.table import (
            DEFAULT_WRITER_TARGET_SIZE_BYTES,
            iceberg_table_properties,
        )
        from flare.data.contracts.publication import PublicationIdentity
        from flare.data.contracts.sink import SinkResult

        fake_ds = MagicMock()
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())])
        table = MagicMock()
        table.properties = iceberg_table_properties(DEFAULT_WRITER_TARGET_SIZE_BYTES)
        snapshot = MagicMock()
        snapshot.snapshot_id = 1
        snapshot.summary = {
            "added-records": "123",
            "flare.publication-id": "abc",
        }
        written_table = MagicMock()
        written_table.current_snapshot.return_value = snapshot

        with (
            patch(
                "flare.data.connectors.adapters.iceberg.sink.IcebergTableSession",
                return_value=SimpleNamespace(table=lambda: table, close=lambda: None),
            ),
            patch(
                "flare.data.connectors.adapters.iceberg.observation.IcebergTableSession",
                return_value=SimpleNamespace(
                    table=lambda: written_table, close=lambda: None
                ),
            ),
            patch(
                "flare.data.connectors.adapters.iceberg.sink.validate_iceberg_table"
            ) as validate_table,
            patch(
                "flare.data.connectors.adapters.iceberg.sink.FixedSchemaAppendIcebergDatasink"
            ) as datasink_type,
        ):
            consumer = IcebergSnapshotPublisher(
                table_identifier="db.tbl",
                catalog_provider=ConfiguredIcebergCatalogProvider(
                    {
                        "name": "rest",
                        "type": "rest",
                        "uri": "http://iceberg-rest.local",
                    }
                ),
                write_task_num_cpus=0.5,
                write_concurrency=3,
            )
            result = consumer.publish(
                fake_ds,
                schema=schema,
                publication=PublicationIdentity("abc"),
            )

        assert isinstance(result, SinkResult)
        assert result.total_rows == 123
        fake_ds.repartition.assert_not_called()
        fake_ds.map_batches.assert_not_called()
        fake_ds.schema.assert_not_called()
        datasink_type.assert_called_once()
        datasink_kwargs = datasink_type.call_args.kwargs
        assert datasink_kwargs == {
            "coalesce_target_bytes": DEFAULT_WRITER_TARGET_SIZE_BYTES,
            "min_rows_per_write": None,
            "expected_schema": schema,
            "expected_iceberg_schema": validate_table.return_value,
            "table_identifier": "db.tbl",
            "catalog_provider": consumer._catalog_provider,
            "snapshot_properties": {"flare.publication-id": "abc"},
        }
        fake_ds.write_datasink.assert_called_once_with(
            datasink_type.return_value,
            ray_remote_args={"num_cpus": 0.5},
            concurrency=3,
        )
        fake_ds.write_iceberg.assert_not_called()
        fake_ds.count.assert_not_called()

    def test_iceberg_consumer_default_write_concurrency_is_ray_dynamic(self):
        from unittest.mock import MagicMock, patch

        import pyarrow as pa

        from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher
        from flare.data.connectors.adapters.iceberg.table import (
            DEFAULT_WRITER_TARGET_SIZE_BYTES,
            iceberg_table_properties,
        )

        fake_ds = MagicMock()
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())])
        table = MagicMock()
        table.properties = iceberg_table_properties(DEFAULT_WRITER_TARGET_SIZE_BYTES)
        snapshot = MagicMock()
        snapshot.snapshot_id = 1
        snapshot.summary = {
            "added-records": "123",
            "flare.publication-id": "abc",
        }
        written_table = MagicMock()
        written_table.current_snapshot.return_value = snapshot

        with (
            patch(
                "flare.data.connectors.adapters.iceberg.sink.IcebergTableSession",
                return_value=SimpleNamespace(table=lambda: table, close=lambda: None),
            ),
            patch(
                "flare.data.connectors.adapters.iceberg.observation.IcebergTableSession",
                return_value=SimpleNamespace(
                    table=lambda: written_table, close=lambda: None
                ),
            ),
            patch("flare.data.connectors.adapters.iceberg.sink.validate_iceberg_table"),
            patch(
                "flare.data.connectors.adapters.iceberg.sink.FixedSchemaAppendIcebergDatasink"
            ) as datasink_type,
        ):
            IcebergSnapshotPublisher(
                table_identifier="db.tbl",
                catalog_provider=ConfiguredIcebergCatalogProvider(
                    {
                        "name": "rest",
                        "type": "rest",
                        "uri": "http://iceberg-rest.local",
                    }
                ),
                write_task_num_cpus=1.0,
            ).publish(
                fake_ds,
                schema=schema,
                publication=PublicationIdentity("abc"),
            )
            fake_ds.write_datasink.assert_called_once_with(
                datasink_type.return_value,
                ray_remote_args={"num_cpus": 1.0},
                concurrency=None,
            )
            fake_ds.write_iceberg.assert_not_called()
            fake_ds.repartition.assert_not_called()
            fake_ds.map_batches.assert_not_called()
            fake_ds.schema.assert_not_called()
            fake_ds.count.assert_not_called()

    def test_iceberg_consumer_adds_publication_id_without_mutating_properties(self):
        from unittest.mock import MagicMock, patch

        import pyarrow as pa

        from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher
        from flare.data.connectors.adapters.iceberg.table import (
            DEFAULT_WRITER_TARGET_SIZE_BYTES,
            iceberg_table_properties,
        )

        fake_ds = MagicMock()
        schema = pa.schema([pa.field("id", pa.int64())])
        table = MagicMock()
        table.properties = iceberg_table_properties(DEFAULT_WRITER_TARGET_SIZE_BYTES)
        snapshot = MagicMock()
        snapshot.snapshot_id = 1
        snapshot.summary = {
            "added-records": "1",
            "flare.publication-id": "abc",
        }
        written_table = MagicMock()
        written_table.current_snapshot.return_value = snapshot
        properties = {"experiment": "resume-v1"}

        with (
            patch(
                "flare.data.connectors.adapters.iceberg.sink.IcebergTableSession",
                return_value=SimpleNamespace(table=lambda: table, close=lambda: None),
            ),
            patch(
                "flare.data.connectors.adapters.iceberg.observation.IcebergTableSession",
                return_value=SimpleNamespace(
                    table=lambda: written_table, close=lambda: None
                ),
            ),
            patch("flare.data.connectors.adapters.iceberg.sink.validate_iceberg_table"),
            patch(
                "flare.data.connectors.adapters.iceberg.sink.FixedSchemaAppendIcebergDatasink"
            ) as datasink_type,
        ):
            IcebergSnapshotPublisher(
                table_identifier="db.tbl",
                catalog_provider=ConfiguredIcebergCatalogProvider(
                    {
                        "name": "rest",
                        "type": "rest",
                        "uri": "http://iceberg-rest.local",
                    }
                ),
                write_task_num_cpus=1.0,
                snapshot_properties=properties,
            ).publish(
                fake_ds,
                schema=schema,
                publication=PublicationIdentity("abc"),
            )

        assert properties == {"experiment": "resume-v1"}
        assert datasink_type.call_args.kwargs["snapshot_properties"] == {
            "experiment": "resume-v1",
            "flare.publication-id": "abc",
        }

    def test_iceberg_consumer_rejects_current_snapshot_from_other_publication(self):
        from unittest.mock import MagicMock, patch

        import pyarrow as pa

        from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher
        from flare.data.connectors.adapters.iceberg.table import (
            DEFAULT_WRITER_TARGET_SIZE_BYTES,
            iceberg_table_properties,
        )

        fake_ds = MagicMock()
        schema = pa.schema([pa.field("id", pa.int64())])
        table = MagicMock()
        table.properties = iceberg_table_properties(DEFAULT_WRITER_TARGET_SIZE_BYTES)
        snapshot = MagicMock()
        snapshot.snapshot_id = 1
        snapshot.summary = {
            "added-records": "1",
            "flare.publication-id": "other",
        }
        written_table = MagicMock()
        written_table.current_snapshot.return_value = snapshot

        with (
            patch(
                "flare.data.connectors.adapters.iceberg.sink.IcebergTableSession",
                return_value=SimpleNamespace(table=lambda: table, close=lambda: None),
            ),
            patch(
                "flare.data.connectors.adapters.iceberg.observation.IcebergTableSession",
                return_value=SimpleNamespace(
                    table=lambda: written_table, close=lambda: None
                ),
            ),
            patch("flare.data.connectors.adapters.iceberg.sink.validate_iceberg_table"),
            pytest.raises(RuntimeError, match="publication id mismatch"),
        ):
            IcebergSnapshotPublisher(
                table_identifier="db.tbl",
                catalog_provider=ConfiguredIcebergCatalogProvider(
                    {
                        "name": "rest",
                        "type": "rest",
                        "uri": "http://iceberg-rest.local",
                    }
                ),
                write_task_num_cpus=1.0,
            ).publish(
                fake_ds,
                schema=schema,
                publication=PublicationIdentity("abc"),
            )

    def test_iceberg_consumer_requires_catalog(self):
        from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher

        with pytest.raises(ValueError):
            IcebergSnapshotPublisher(
                table_identifier="db.tbl",
                catalog_provider=ConfiguredIcebergCatalogProvider({}),
                write_task_num_cpus=1.0,
            )

    def test_iceberg_publisher_requires_an_existing_table(self):
        from unittest.mock import patch

        import pyiceberg.exceptions

        from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher

        with (
            patch(
                "flare.data.connectors.adapters.iceberg.sink.IcebergTableSession",
                side_effect=pyiceberg.exceptions.NoSuchTableError("db.tbl"),
            ),
            pytest.raises(pyiceberg.exceptions.NoSuchTableError),
        ):
            IcebergSnapshotPublisher(
                table_identifier="db.tbl",
                catalog_provider=ConfiguredIcebergCatalogProvider(
                    {
                        "name": "rest",
                        "type": "rest",
                        "uri": "http://iceberg-rest.local",
                    }
                ),
                write_task_num_cpus=1.0,
            )

    def test_inflight_consumer_drains_stream(self):
        """RayDataStreamDrainer.consume must drain the dataset via ``iter_batches()``
        to avoid pinning all blocks in plasma simultaneously."""
        from unittest.mock import MagicMock

        import pyarrow as pa

        from flare.data.connectors.adapters.stream.sink import RayDataStreamDrainer

        batch = MagicMock(spec=pa.Table)
        batch.num_rows = 1234
        fake_ds = MagicMock()
        fake_ds.iter_batches.return_value = [batch]

        consumer = RayDataStreamDrainer(output_format="arrow", prefetch_batches=2)
        assert consumer.dataset is None
        result = consumer.finalize(fake_ds, schema=pa.schema([]))

        # streaming-drain contract: must call iter_batches
        fake_ds.iter_batches.assert_called_once_with(
            batch_size=None,
            batch_format="pyarrow",
            prefetch_batches=2,
        )
        assert not fake_ds.materialize.called
        assert not fake_ds.write_datasink.called
        assert consumer.dataset is None
        assert result.total_rows == 1234

    def test_orchestrator_execute_signature_takes_consumer(self):
        """OCP entry point: ``execute(file_paths, consumer)`` must remain stable.

        Also pins down the constructor signature: orchestrator no longer
        owns a ``dataset_uri`` / ``storage_options`` — those belong to the
        injected consumer.
        """
        import inspect

        from flare.data.pipeline.production.publish.orchestrator import RayDataPipeline

        execute_params = inspect.signature(RayDataPipeline.execute).parameters
        assert list(execute_params)[1:] == ["file_paths", "terminal"]

        init_params = inspect.signature(RayDataPipeline.__init__).parameters
        assert "dataset_uri" not in init_params
        assert "storage_options" not in init_params

    def test_orchestrator_passes_compiled_plan_and_resources_to_workers(self):
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        import pyarrow as pa
        import ray

        from flare.data.pipeline.production.publish.config import PipelineConfig
        from flare.data.pipeline.production.publish.orchestrator import RayDataPipeline

        config = PipelineConfig.create(
            cluster_cpus=8,
            cluster_memory_bytes=16 * 1024**3,
            object_store_capacity_bytes=4 * 1024**3,
            read_task_memory_bytes=512 * 1024**2,
            n_gpus=1,
            gpu_memory_bytes=16 * 1024**3,
            gpu_mode="force",
            encoders_per_gpu=2,
        )
        fake_artifact = MagicMock()
        fake_artifact.compiled_pipeline.output_schema = pa.schema(
            [pa.field("x", pa.int64())]
        )
        fake_artifact.compiled_pipeline.digest = "a" * 64
        fake_artifact.parser_spec = object()
        fake_artifact.metadata_maps = {"kind": {"x": 1}}
        fake_artifact.encoded_plan = b"plan"
        fake_consumer = MagicMock()
        fake_consumer.finalize.return_value.total_rows = 0
        fake_ds = MagicMock()
        fake_ds.map_batches.return_value = fake_ds

        with (
            patch(
                "flare.data.pipeline.production.publish.orchestrator.ProcessingArtifactCompiler"
            ) as compiler,
            patch(
                "flare.data.pipeline.production.publish.orchestrator.estimate_arrow_row_bytes",
                return_value=8,
            ),
            patch(
                "flare.data.pipeline.production.publish.orchestrator.HdfsZstdDatasource"
            ) as datasource_cls,
            patch(
                "flare.data.pipeline.production.publish.orchestrator.ray.data.read_datasource",
                return_value=fake_ds,
            ) as read_datasource,
        ):
            compiler.return_value.compile.return_value = fake_artifact
            pipeline = RayDataPipeline(
                config=config,
                conf_dir=str(Path("/tmp/conf")),
                model_conf_dir=str(Path("/tmp/model")),
                parser_contract=object(),
            )
            pipeline.execute(("/tmp/part-00000.zst",), fake_consumer)

        fake_consumer.finalize.assert_called_once_with(
            fake_ds,
            schema=pipeline.output_schema,
        )
        datasource_cls.assert_called_once_with(
            source_partition_defaults=config.source_partition_defaults,
            source_provider=None,
            file_paths=("/tmp/part-00000.zst",),
            source_csv_block_size_bytes=config.source_csv_block_size_bytes,
            read_planner_block_bytes=config.read_planner_block_bytes,
            datasource_prefetch_max_buffers=config.datasource_prefetch_max_buffers,
            datasource_prefetch_buffer_size_bytes=(
                config.datasource_prefetch_buffer_size_bytes
            ),
            source_files_per_task=config.source_files_per_task,
            read_profile=config.read_profile,
            read_profile_collector=None,
        )
        context = ray.data.DataContext.get_current()
        assert context.downstream_capacity_backpressure_ratio == (
            config.downstream_capacity_backpressure_ratio
        )
        assert (
            context.enable_dynamic_output_queue_size_backpressure
            is config.dynamic_output_queue_backpressure
        )
        assert (
            context._max_num_blocks_in_streaming_gen_buffer
            == config.streaming_generator_max_blocks
        )
        assert context.eager_free is config.eager_free

        encode_call = fake_ds.map_batches.call_args_list[1]
        parse_call = fake_ds.map_batches.call_args_list[0]
        assert encode_call.kwargs["fn_constructor_kwargs"] == {
            "encoded_plan": b"plan",
            "output_schema": fake_artifact.compiled_pipeline.output_schema,
        }
        assert parse_call.kwargs["fn_constructor_kwargs"] == {
            "parser_spec": fake_artifact.parser_spec,
            "metadata_maps": {"kind": {"x": 1}},
        }
        assert encode_call.kwargs["memory"] == config.encode_actor_memory_bytes
        assert (
            parse_call.kwargs["compute"].max_tasks_in_flight_per_actor
            == config.actor_max_tasks_in_flight
        )
        assert (
            encode_call.kwargs["compute"].max_tasks_in_flight_per_actor
            == config.actor_max_tasks_in_flight
        )
        assert parse_call.kwargs["memory"] == config.parse_actor_memory_bytes
        assert parse_call.kwargs["batch_size"] == pipeline._max_rows_per_batch
        assert encode_call.kwargs["batch_size"] == pipeline._max_rows_per_batch
        read_call = read_datasource.call_args
        assert read_call.kwargs["override_num_blocks"] == 1

    def test_orchestrator_fuses_parse_and_encode_with_conserved_resources(self):
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        import pyarrow as pa

        from flare.data.pipeline.production.publish.config import PipelineConfig
        from flare.data.pipeline.production.publish.orchestrator import RayDataPipeline

        config = PipelineConfig.create(
            cluster_cpus=8,
            cluster_memory_bytes=16 * 1024**3,
            object_store_capacity_bytes=4 * 1024**3,
            read_task_memory_bytes=512 * 1024**2,
            n_gpus=1,
            gpu_memory_bytes=16 * 1024**3,
            gpu_mode="force",
            encoders_per_gpu=2,
            fuse_parse_encode=True,
        )
        artifact = MagicMock()
        artifact.compiled_pipeline.output_schema = pa.schema(
            [pa.field("x", pa.int64())]
        )
        artifact.compiled_pipeline.digest = "a" * 64
        artifact.parser_spec = object()
        artifact.metadata_maps = {"kind": {"x": 1}}
        artifact.encoded_plan = b"plan"
        consumer = MagicMock()
        consumer.finalize.return_value.total_rows = 0
        ds = MagicMock()
        ds.map_batches.return_value = ds

        with (
            patch(
                "flare.data.pipeline.production.publish.orchestrator.ProcessingArtifactCompiler"
            ) as compiler,
            patch(
                "flare.data.pipeline.production.publish.orchestrator.estimate_arrow_row_bytes",
                return_value=8,
            ),
            patch(
                "flare.data.pipeline.production.publish.orchestrator.HdfsZstdDatasource"
            ),
            patch(
                "flare.data.pipeline.production.publish.orchestrator.ray.data.read_datasource",
                return_value=ds,
            ),
        ):
            compiler.return_value.compile.return_value = artifact
            pipeline = RayDataPipeline(
                config=config,
                conf_dir=str(Path("/tmp/conf")),
                model_conf_dir=str(Path("/tmp/model")),
                parser_contract=object(),
            )
            pipeline.execute(("/tmp/part-00000.zst",), consumer)

        assert ds.map_batches.call_count == 1
        call = ds.map_batches.call_args
        assert call.args[0].__name__ == "FusedParseEncodeUDF"
        assert call.kwargs["fn_constructor_kwargs"] == {
            "parser_spec": artifact.parser_spec,
            "metadata_maps": {"kind": {"x": 1}},
            "encoded_plan": b"plan",
            "output_schema": artifact.compiled_pipeline.output_schema,
        }
        assert call.kwargs["memory"] == (
            config.parse_actor_memory_bytes + config.encode_actor_memory_bytes
        )
        strategy = call.kwargs["compute"]
        assert strategy.min_size == (
            config.parse_actor_min_size + config.encode_actor_min_size
        )
        assert strategy.max_size == (
            config.parse_actor_max_size + config.encode_actor_max_size
        )
