import io
from contextlib import contextmanager
from types import SimpleNamespace

import pyarrow as pa

from flare.data.pipeline.production.publish.datasource import HdfsZstdDatasource
from flare.data.pipeline.production.publish.runner import discover_publication_manifest


def test_source_provider_handles_discovery_and_streaming_without_native_transport(
    monkeypatch,
):
    path = "memory://fixture/ds=2000-01-01/part-0.zst"
    payload = pa.compress(b"first\nsecond\n", codec="zstd").to_pybytes()
    events = []

    class Session:
        root = "memory://fixture"

        def list_directory(self, directory):
            assert directory == self.root + "/ds=2000-01-01"
            return [path]

        def open(self, name):
            assert name == path
            return io.BytesIO(payload)

    class Provider:
        @contextmanager
        def open(self):
            events.append("open")
            try:
                yield Session()
            finally:
                events.append("close")

    def reject_native_open(*args):
        raise AssertionError("Injected source must own its transport")

    monkeypatch.setattr(
        "flare.data.pipeline.production.publish.datasource.open_hdfs_file",
        reject_native_open,
    )
    provider = Provider()
    manifest = discover_publication_manifest(
        SimpleNamespace(
            start_date="2000-01-01", end_date="2000-01-01", limit_files=None
        ),
        SimpleNamespace(is_pending=lambda path: True),
        provider,
    )
    assert manifest.files == (path,)
    source = HdfsZstdDatasource(
        manifest.files,
        source_provider=provider,
        source_csv_block_size_bytes=1024,
        read_planner_block_bytes=1024,
        datasource_prefetch_max_buffers=1,
        datasource_prefetch_buffer_size_bytes=1024,
        source_files_per_task=1,
        source_partition_defaults={"_hour": "daily"},
    )
    result = pa.concat_tables(list(source.get_read_tasks(1)[0]()))
    assert result.to_pydict() == {
        "line": ["first", "second"],
        "_ds": ["2000-01-01"] * 2,
        "_hour": ["daily"] * 2,
    }
    assert events == ["open", "close", "open", "close"]
