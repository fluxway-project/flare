"""Hard-exit recovery against a real SQLite Iceberg catalog and Parquet files."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pyiceberg.catalog.sql import SqlCatalog

import flare
from flare.data.connectors.adapters.iceberg.table import create_iceberg_table
from flare.data.pipeline.production.publication.run_lock import publication_run_lock
from flare.data.pipeline.production.publish.tests.crash_recovery_worker import (
    example_table,
)


def _invoke(root: Path, mode: str):
    script = Path(__file__).with_name("crash_recovery_worker.py")
    env = os.environ.copy()
    source_root = Path(flare.__file__).resolve().parent.parent
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), env.get("PYTHONPATH")))
    )
    return subprocess.run(
        [sys.executable, str(script), str(root), mode],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


@pytest.mark.parametrize("crash", ["before_commit", "after_commit"])
def test_hard_exit_then_resume_commits_each_source_once(tmp_path, crash):
    import json

    options = {
        "name": "recovery",
        "type": "sql",
        "uri": f"sqlite:///{tmp_path / 'catalog.db'}",
        "warehouse": tmp_path.as_uri(),
    }
    (tmp_path / "catalog.json").write_text(json.dumps(options))
    catalog = SqlCatalog(**options)
    catalog.create_namespace("test")
    table = create_iceberg_table(
        catalog=catalog,
        table_identifier="test.events",
        schema=example_table().schema,
        writer_target_size_bytes=1024 * 1024,
    )
    first = _invoke(tmp_path, crash)
    assert first.returncode == (85 if crash == "before_commit" else 86), first.stderr
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["processed"] == [] and len(state["in_flight"]) == 1
    table.refresh()
    assert len(table.snapshots()) == (0 if crash == "before_commit" else 1)
    for mode in ["normal", "normal", "repeat"]:
        result = _invoke(tmp_path, mode)
        assert result.returncode == 0, result.stderr
    table.refresh()
    assert len(table.snapshots()) == 1
    assert table.scan().to_arrow()["id"].to_pylist() == [1, 2, 3]
    assert int(table.current_snapshot().summary["total-records"]) == 3
    state = json.loads((tmp_path / "state.json").read_text())
    assert sorted(state["processed"]) == ["source-a", "source-b"]
    assert state["in_flight"] == {}
    catalog.close()


def test_another_process_cannot_take_live_resume_lock(tmp_path):
    state = str(tmp_path / "state.json")
    with publication_run_lock(state):
        result = _invoke(tmp_path, "lock")
        assert result.returncode != 0
        assert "Another producer holds the resume lock" in result.stderr
    result = _invoke(tmp_path, "lock")
    assert result.returncode == 0, result.stderr
