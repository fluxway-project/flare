import pytest

from flare.data.connectors.adapters.iceberg.scan_cursor import (
    IcebergScanCursor,
    ManifestEntryOffset,
    ManifestListRef,
    ManifestRef,
)
from flare.data.connectors.adapters.iceberg.scan_result import (
    FileScanRange,
    FileScanResult,
)


def _cursor() -> IcebergScanCursor:
    return IcebergScanCursor.create(
        snapshot_id=7,
        contract_digest="a" * 64,
        manifest_list_ref=ManifestListRef.create(
            snapshot_id=7, path="list.avro", object_length=1
        ),
    )


def _range(count):
    cursor = _cursor()
    return FileScanRange(
        "a.parquet",
        count,
        "parquet",
        IcebergScanCursor(
            cursor.state.with_manifest(
                ManifestRef.create(
                    path="manifest.avro",
                    object_length=1,
                    content="0",
                    sequence=1,
                    added_snapshot_id=7,
                    partition_spec_id=0,
                ),
                ManifestEntryOffset(0, 0, 0),
            )
        ),
    )


def test_scan_result_is_bounded_value() -> None:
    cursor = _cursor()
    scan_result = FileScanResult(
        files=(_range(3),),
        next_cursor=cursor,
        snapshot_exhausted=False,
    )
    assert scan_result.files[0].record_count == 3
    assert scan_result.record_count == 3
    scan_result.validate_budget(max_records=3)
    terminal_scan_result = FileScanResult(
        files=scan_result.files,
        next_cursor=cursor,
        snapshot_exhausted=True,
    )
    terminal_scan_result.validate_transition(cursor)


def test_ready_scan_result_must_advance_cursor() -> None:
    requested = _cursor()
    scan_result = FileScanResult(
        files=(_range(3),),
        next_cursor=requested,
        snapshot_exhausted=False,
    )
    with pytest.raises(ValueError, match="did not advance"):
        scan_result.validate_transition(requested)


def test_ready_scan_result_rejects_cursor_regression() -> None:
    requested = _cursor()
    regressed = _cursor()
    scan_result = FileScanResult(
        files=(_range(1),),
        next_cursor=regressed,
        snapshot_exhausted=False,
    )
    with pytest.raises(ValueError, match="did not advance"):
        scan_result.validate_transition(requested)


def test_scan_result_budget_is_owned_by_scan_result() -> None:
    cursor = _cursor()
    scan_result = FileScanResult(
        files=(_range(3),),
        next_cursor=cursor,
        snapshot_exhausted=False,
    )
    with pytest.raises(ValueError, match="record budget"):
        scan_result.validate_budget(max_records=2)
    other = _range(1)
    bounded = FileScanResult(
        files=(scan_result.files[0], other),
        next_cursor=cursor,
        snapshot_exhausted=False,
    )
    with pytest.raises(ValueError, match="unit budget"):
        bounded.validate_budget(max_records=4, max_units=1)


def test_scan_result_rejects_invalid_file_fact() -> None:
    cursor = _cursor()
    with pytest.raises(ValueError, match="record count"):
        FileScanResult(
            files=(_range(-1),),
            next_cursor=cursor,
            snapshot_exhausted=False,
        )


def test_snapshot_exhaustion_requires_a_boolean() -> None:
    with pytest.raises(TypeError, match="snapshot_exhausted"):
        FileScanResult(files=(), next_cursor=_cursor(), snapshot_exhausted=1)
