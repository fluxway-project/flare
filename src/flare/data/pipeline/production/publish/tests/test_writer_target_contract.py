from types import SimpleNamespace

from flare.data.connectors.adapters.iceberg.table import (
    DEFAULT_WRITER_TARGET_SIZE_BYTES,
)
from flare.data.pipeline.production.publish.runner import (
    _resolve_writer_target_size_bytes,
)


def test_default_iceberg_writer_target_is_independent_of_ray_block_size() -> None:
    args = SimpleNamespace(iceberg_writer_target_size_mib=None)

    assert _resolve_writer_target_size_bytes(args) == DEFAULT_WRITER_TARGET_SIZE_BYTES


def test_explicit_iceberg_writer_target_overrides_table_default() -> None:
    args = SimpleNamespace(iceberg_writer_target_size_mib=256)

    assert _resolve_writer_target_size_bytes(args) == 256 * 1024 * 1024


def test_explicit_iceberg_writer_target_must_be_positive() -> None:
    args = SimpleNamespace(iceberg_writer_target_size_mib=0)

    try:
        _resolve_writer_target_size_bytes(args)
    except ValueError as error:
        assert "must be positive" in str(error)
    else:
        raise AssertionError("non-positive Iceberg writer target was accepted")
