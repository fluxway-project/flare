from pathlib import Path

import pytest

from flare.data.pipeline.production.builder.hdfs_source import (
    HdfsPartitionCursor,
    observe_hdfs_partition,
)


def test_success_marker_makes_partition_complete_without_opening_files(tmp_path: Path):
    partition = tmp_path / "ds=2026-07-20"
    hour = partition / "hour=00"
    hour.mkdir(parents=True)
    (hour / "part-0.zst").write_bytes(b"metadata-only")
    (partition / "_SUCCESS").write_bytes(b"")

    files, cursor, complete = observe_hdfs_partition(str(partition), now_ms=100)
    assert complete
    assert [item.path.rsplit("/", 1)[-1] for item in files] == [
        "_SUCCESS",
        "part-0.zst",
    ]
    restored = HdfsPartitionCursor.from_checkpoint(cursor.checkpoint_state())
    assert restored.fingerprint == cursor.fingerprint


def test_stability_window_is_explicit_and_length_mutation_breaks_stability(
    tmp_path: Path,
):
    partition = tmp_path / "ds=2026-07-20"
    hour = partition / "hour=00"
    hour.mkdir(parents=True)
    part = hour / "part-0.zst"
    part.write_bytes(b"one")

    _files, cursor, complete = observe_hdfs_partition(
        str(partition), now_ms=100, stability_window_s=10
    )
    assert not complete
    _, stable_cursor, complete = observe_hdfs_partition(
        str(partition), now_ms=10_100, cursor=cursor, stability_window_s=10
    )
    assert complete
    part.write_bytes(b"changed")
    _, changed_cursor, complete = observe_hdfs_partition(
        str(partition), now_ms=20_100, cursor=stable_cursor, stability_window_s=10
    )
    assert not complete
    assert changed_cursor.fingerprint != stable_cursor.fingerprint


def test_stability_window_keeps_first_unchanged_observation(tmp_path: Path):
    partition = tmp_path / "ds=2026-07-20"
    partition.mkdir()
    (partition / "part-0.zst").write_bytes(b"one")

    _, first, complete = observe_hdfs_partition(
        str(partition), now_ms=100, stability_window_s=10
    )
    assert not complete
    _, middle, complete = observe_hdfs_partition(
        str(partition), now_ms=5_100, cursor=first, stability_window_s=10
    )
    assert not complete
    assert middle.observed_at_ms == 100
    _, final, complete = observe_hdfs_partition(
        str(partition), now_ms=10_100, cursor=middle, stability_window_s=10
    )
    assert complete
    assert final.observed_at_ms == 100


def test_partition_cursor_rejects_missing_checkpoint_fields() -> None:
    state = HdfsPartitionCursor(
        partition="hdfs://example/data/pipeline/ds=2026-07-20"
    ).checkpoint_state()
    del state["published_paths"]

    with pytest.raises(ValueError, match="fields do not match"):
        HdfsPartitionCursor.from_checkpoint(state)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("partition", 7),
        ("observed_at_ms", "100"),
        ("published_paths", [7]),
    ),
)
def test_partition_cursor_rejects_coerced_checkpoint_fields(field, value) -> None:
    state = HdfsPartitionCursor(
        partition="hdfs://example/data/pipeline/ds=2026-07-20"
    ).checkpoint_state()
    state[field] = value

    with pytest.raises((TypeError, ValueError)):
        HdfsPartitionCursor.from_checkpoint(state)


def test_partition_cursor_rejects_coerced_fingerprint_fields() -> None:
    state = HdfsPartitionCursor(
        partition="hdfs://example/data/pipeline/ds=2026-07-20",
        fingerprint=(("part.zst", 10, 100, False),),
    ).checkpoint_state()
    fingerprint = state["fingerprint"]
    if not isinstance(fingerprint, list):
        raise TypeError("checkpoint fingerprint must be a list")
    fingerprint[0][3] = "false"

    with pytest.raises((TypeError, ValueError)):
        HdfsPartitionCursor.from_checkpoint(state)
