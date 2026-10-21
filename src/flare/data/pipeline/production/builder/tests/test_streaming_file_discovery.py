from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from flare.data.pipeline.production.builder.hdfs_source import (
    hdfs_ls_parallel_iter,
    iter_hdfs_file_paths,
)
from flare.data.pipeline.production.publication.manifest import PublicationManifest


class _PendingState:
    def is_pending(self, file_path: str) -> bool:
        return not file_path.endswith("2.zst")


def test_local_discovery_is_lazy(tmp_path: Path) -> None:
    today = datetime.now(timezone.utc).date()
    day = today.isoformat()
    hour = tmp_path / f"ds={day}" / "hour=00"
    hour.mkdir(parents=True)
    for index in range(4):
        (hour / f"part-{index}.zst").write_bytes(b"")

    discovered = iter_hdfs_file_paths(str(tmp_path), today, today)
    assert iter(discovered) is discovered
    assert next(discovered).endswith("part-0.zst")
    assert list(discovered) == [
        str(hour / "part-1.zst"),
        str(hour / "part-2.zst"),
        str(hour / "part-3.zst"),
    ]


def test_pending_manifest_filters_completed_files() -> None:
    paths = (f"part-{index}.zst" for index in range(7))
    manifest = PublicationManifest.from_files(
        path for path in paths if _PendingState().is_pending(path)
    )
    assert manifest.files == (
        "part-0.zst",
        "part-1.zst",
        "part-3.zst",
        "part-4.zst",
        "part-5.zst",
        "part-6.zst",
    )


def test_parallel_hdfs_listing_preserves_directory_and_entry_order() -> None:
    delays = {"hour=00": 0.02, "hour=01": 0.01, "hour=02": 0.0}

    def list_directory(directory: str) -> list[str]:
        time.sleep(delays[directory])
        return [f"{directory}/part-1.zst", f"{directory}/part-0.zst"]

    with patch(
        "flare.data.pipeline.production.builder.hdfs_source.hdfs_ls",
        side_effect=list_directory,
    ):
        paths = list(
            hdfs_ls_parallel_iter(
                ["hour=00", "hour=01", "hour=02"],
                max_workers=3,
            )
        )

    assert paths == [
        f"hour={hour}/part-{part}.zst" for hour in ("00", "01", "02") for part in (0, 1)
    ]
