from __future__ import annotations

import json

import pytest

from flare.data.connectors.adapters.iceberg.scan_cursor import (
    IcebergScanCursor,
    ManifestEntryOffset,
    ManifestListOffset,
    ManifestListRef,
    ManifestRef,
    TraversalState,
)


def _cursor() -> IcebergScanCursor:
    list_ref = ManifestListRef.create(
        snapshot_id=17, path="list.avro", object_length=1024
    )
    return IcebergScanCursor.create(
        snapshot_id=17, contract_digest="a" * 64, manifest_list_ref=list_ref
    )


def test_iceberg_scan_cursor_round_trips_typed_state() -> None:
    cursor = _cursor()
    manifest = ManifestRef.create(
        path="manifest.avro",
        object_length=256,
        content="0",
        sequence=3,
        added_snapshot_id=17,
        partition_spec_id=0,
    )
    state = cursor.state.with_manifest(manifest, ManifestEntryOffset(128, 4, 7))
    cursor = IcebergScanCursor(
        TraversalState(
            17,
            "a" * 64,
            cursor.state.manifest_list,
            ManifestListOffset(64, 2, 5),
            manifest,
            state.manifest_entry_offset,
        )
    )
    restored = IcebergScanCursor.decode(cursor.encode())
    assert restored == cursor
    assert restored.encode() == cursor.encode()
    encoded = json.loads(cursor.encode())
    assert "manifest_identity" not in encoded["state"]
    assert "next_range_identity" not in encoded["state"]


def test_iceberg_scan_cursor_rejects_tampering() -> None:
    cursor = _cursor()
    payload = cursor.encode().replace(b'"row_offset":0', b'"row_offset":1')
    with pytest.raises(ValueError, match="digest"):
        IcebergScanCursor.decode(payload)


def test_iceberg_scan_cursor_rejects_unknown_fields() -> None:
    cursor = _cursor()
    payload = cursor.encode()[:-1] + b',"extra":1}'
    with pytest.raises(ValueError, match="fields"):
        IcebergScanCursor.decode(payload)


def test_iceberg_scan_cursor_rejects_legacy_version_field() -> None:
    cursor = _cursor()
    payload = cursor.encode()[:-1] + b',"legacy_field":"old"}'
    with pytest.raises(ValueError, match="fields"):
        IcebergScanCursor.decode(payload)


def test_cursor_traversal_position_is_monotonic_within_manifest() -> None:
    initial = _cursor()
    manifest = ManifestRef.create(
        path="manifest.avro",
        object_length=256,
        content="0",
        sequence=3,
        added_snapshot_id=17,
        partition_spec_id=0,
    )
    prior = IcebergScanCursor(
        initial.state.with_manifest(manifest, ManifestEntryOffset(128, 3, 4))
    )
    following = IcebergScanCursor(
        initial.state.with_manifest(manifest, ManifestEntryOffset(128, 3, 5))
    )
    assert following.is_after(prior)
    assert not prior.is_after(following)


def test_cursor_rejects_mismatched_semantic_ref_and_offset() -> None:
    cursor = _cursor()
    manifest = ManifestRef.create(
        path="manifest.avro",
        object_length=256,
        content="0",
        sequence=3,
        added_snapshot_id=17,
        partition_spec_id=0,
    )
    with pytest.raises(ValueError, match="paired"):
        TraversalState(
            17,
            "a" * 64,
            cursor.state.manifest_list,
            ManifestListOffset(0, 0, 0),
            manifest,
            None,
        )
