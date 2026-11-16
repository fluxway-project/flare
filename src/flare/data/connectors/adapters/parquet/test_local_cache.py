from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
from types import SimpleNamespace

from flare.data.connectors.adapters.parquet.local_cache import LocalParquetFileCache


def test_local_cache_downloads_once_across_concurrent_readers(tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"immutable parquet bytes")
    cache = LocalParquetFileCache(1024, tmp_path / "cache")
    count = 0
    lock = Lock()

    def factory():
        def open_source():
            nonlocal count
            with lock:
                count += 1
            sleep(0.05)
            return source.open("rb")

        return SimpleNamespace(open=open_source)

    def read():
        with cache.open_file(
            revision_identity="snapshot-1",
            locator="s3://bucket/source.parquet",
            input_file_factory=factory,
        ) as cached:
            return cached.read()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: read(), range(4))) == [source.read_bytes()] * 4
    assert count == 1


def test_local_cache_evicts_old_files_without_invalidating_open_handles(tmp_path):
    cache = LocalParquetFileCache(5, tmp_path / "cache")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"12345")
    second.write_bytes(b"1234")
    first_handle = cache.open_file(
        revision_identity="s",
        locator="first",
        input_file_factory=lambda: SimpleNamespace(open=lambda: first.open("rb")),
    )
    try:
        with cache.open_file(
            revision_identity="s",
            locator="second",
            input_file_factory=lambda: SimpleNamespace(open=lambda: second.open("rb")),
        ) as current:
            assert current.read() == b"1234"
        assert first_handle.read() == b"12345"
        assert len(list(cache.directory.glob("*.parquet"))) == 1
    finally:
        first_handle.close()


def test_local_cache_does_not_retain_a_file_larger_than_its_capacity(tmp_path):
    cache = LocalParquetFileCache(5, tmp_path / "cache")
    source = tmp_path / "large"
    source.write_bytes(b"larger than cache")
    with cache.open_file(
        revision_identity="s",
        locator="large",
        input_file_factory=lambda: SimpleNamespace(open=lambda: source.open("rb")),
    ) as handle:
        assert handle.read() == b"larger than cache"
    assert list(cache.directory.glob("*.parquet")) == []
