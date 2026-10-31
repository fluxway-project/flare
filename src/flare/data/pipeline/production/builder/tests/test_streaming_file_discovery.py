from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from flare.data.pipeline.production.builder.hdfs_source import (
    hdfs_ls,
    hdfs_ls_parallel_iter,
    iter_hdfs_file_paths,
    open_hdfs_file,
)
from flare.data.pipeline.production.publication.manifest import PublicationManifest


class _PendingState:
    def is_pending(self, file_path: str) -> bool:
        return not file_path.endswith("2.zst")


def test_local_discovery_is_lazy(tmp_path: Path) -> None:
    today = datetime.now(UTC).date()
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


def test_native_file_uri_listing_preserves_escaping_and_can_be_opened(tmp_path):
    path = tmp_path / "file with spaces.zst"
    path.write_bytes(b"native transport")
    listed = hdfs_ls(tmp_path.as_uri())
    assert listed == [path.as_uri()]
    with open_hdfs_file(listed[0]) as stream:
        assert stream.read() == b"native transport"


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


def test_daily_discovery_does_not_list_files_as_directories(tmp_path: Path) -> None:
    day = datetime.now(UTC).date()
    root = str(tmp_path)
    date_dir = f"{root}/ds={day}"
    entries = {
        date_dir: [
            f"{date_dir}/_SUCCESS",
            f"{date_dir}/part-0.zst",
            f"{date_dir}/hour=01",
        ],
        f"{date_dir}/hour=01": [f"{date_dir}/hour=01/part-1.zst"],
    }
    expected = [f"{date_dir}/part-0.zst", f"{date_dir}/hour=01/part-1.zst"]
    assert (
        list(iter_hdfs_file_paths(root, day, day, list_directory=entries.__getitem__))
        == expected
    )
    for path in expected:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()
    assert list(iter_hdfs_file_paths(root, day, day)) == sorted(expected)


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
