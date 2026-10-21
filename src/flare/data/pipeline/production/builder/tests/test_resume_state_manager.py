import json
from pathlib import Path

import pytest

from flare.data.pipeline.production.builder.hdfs_source import HdfsPartitionCursor
from flare.data.pipeline.production.builder.resume_state_manager import (
    ResumeStateManager,
)
from flare.data.pipeline.production.publication.contract import PublicationProtocol


def test_in_flight_files_are_not_pending(tmp_path: Path) -> None:
    manager = ResumeStateManager(str(tmp_path / "resume.json"))
    manager.mark_in_flight(
        "batch",
        ["part-0.zst"],
        PublicationProtocol.ICEBERG_SNAPSHOT,
    )

    assert not manager.is_pending("part-0.zst")


def test_partition_cursor_round_trips(tmp_path: Path) -> None:
    state_path = str(tmp_path / "resume.json")
    cursor = HdfsPartitionCursor(
        partition="hdfs://example/data/pipeline/ds=2026-07-21",
        fingerprint=(("part-0.zst", 8, 100, False),),
        observed_at_ms=100,
    )
    manager = ResumeStateManager(state_path)
    manager.update_partition_cursor(cursor)

    restored = ResumeStateManager(state_path)
    restored.load()

    assert restored.partition_cursor(cursor.partition) == cursor


def test_block_attempts_share_the_resume_envelope(tmp_path: Path) -> None:
    state_path = str(tmp_path / "resume.json")
    manager = ResumeStateManager(state_path)
    assert manager.record_block_failure("part-0.zst", 129 * 1024**2, 1) == 1

    restored = ResumeStateManager(state_path)
    restored.load()

    assert restored.is_block_quarantined("part-0.zst", 129 * 1024**2)
    assert restored.record_block_failure("part-0.zst", 129 * 1024**2, 2) == 2


def test_resume_envelope_rejects_missing_progress_fields(tmp_path: Path) -> None:
    state_path = tmp_path / "resume.json"
    manager = ResumeStateManager(str(state_path))
    manager.save()
    state = json.loads(state_path.read_text())
    del state["processed"]
    state_path.write_text(json.dumps(state))

    with pytest.raises(ValueError, match="fields do not match"):
        ResumeStateManager(str(state_path)).load()


def test_resume_envelope_round_trips_publication_lease(tmp_path: Path) -> None:
    state_path = str(tmp_path / "resume.json")
    manager = ResumeStateManager(state_path)
    manager.mark_in_flight(
        "publication",
        ["part-0.zst", "part-1.zst"],
        PublicationProtocol.ICEBERG_SNAPSHOT,
    )

    _, in_flight = ResumeStateManager(state_path).load()

    assert in_flight["publication"].files == ("part-0.zst", "part-1.zst")
    assert in_flight["publication"].protocol is PublicationProtocol.ICEBERG_SNAPSHOT


def test_resume_envelope_rejects_obsolete_version(tmp_path: Path) -> None:
    state_path = tmp_path / "resume.json"
    manager = ResumeStateManager(str(state_path))
    manager.save()
    state = json.loads(state_path.read_text())
    state["version"] = 5
    state_path.write_text(json.dumps(state))

    with pytest.raises(ValueError, match="version must be 1"):
        ResumeStateManager(str(state_path)).load()
