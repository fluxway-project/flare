import hashlib
import io
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from recipes.recsys.data import heap_stream


def test_worker_plan_uses_cluster_capacity_without_fixed_cluster_size():
    assert (
        heap_stream.code_sha256()
        == hashlib.sha256(Path(heap_stream.__file__).read_bytes()).hexdigest()
    )
    options = {
        "cluster_cpus": 332,
        "cluster_memory_bytes": 3 * 1024**4,
        "cpus_per_worker": 2,
        "memory_per_worker_bytes": 16 * 1024**3,
    }
    assert heap_stream.plan_heap_workers(**options, requested=None) == 160
    with pytest.raises(ValueError, match="exceeds available capacity"):
        heap_stream.plan_heap_workers(**options, requested=161)


def test_heap_stream_preserves_daily_partition_and_bounded_row_batches(monkeypatch):
    path = "/fixture/ds=2026-01-01/part-0.zst"
    payload = pa.compress(b"a\nb\nc\nd\ne\n", codec="zstd").to_pybytes()

    class Source:
        @contextmanager
        def open(self):
            yield SimpleNamespace(open=lambda name: io.BytesIO(payload))

    class Sink:
        def __init__(self):
            self.batches = []

        def write(self, blocks, context):
            self.batches = list(blocks)
            return SimpleNamespace(
                data_files=(
                    SimpleNamespace(record_count=sum(b.num_rows for b in self.batches)),
                )
            )

    monkeypatch.setattr(
        heap_stream, "FusedParseEncodeUDF", lambda **kwargs: lambda batch: batch
    )
    monkeypatch.setattr(
        heap_stream,
        "default_prefetch_reader_factory",
        lambda name, *, source, **kwargs: source.open(name),
    )
    monkeypatch.setattr(
        heap_stream.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "fixture-node"),
    )
    config = SimpleNamespace(
        datasource_prefetch_buffer_size_bytes=128,
        datasource_prefetch_max_buffers=1,
        source_csv_block_size_bytes=128,
        source_partition_defaults={"_hour": "unknown"},
    )
    sink = Sink()
    _, counts, _ = heap_stream.stream_group(
        (path,), {}, config, Source(), sink, max_rows_per_batch=2
    )
    assert counts == {path: 5}
    assert [batch.num_rows for batch in sink.batches] == [2, 2, 1]
    assert pa.concat_tables(sink.batches).to_pydict() == {
        "line": ["a", "b", "c", "d", "e"],
        "_ds": ["2026-01-01"] * 5,
        "_hour": ["unknown"] * 5,
    }
