from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_preflight_verifies_installed_owner_and_source_boundaries() -> None:
    root = Path(__file__).parents[2]
    command = [
        sys.executable,
        str(root / "tools/validation/eval_weight_refactor_preflight.py"),
        "--flare",
        str(root),
        "--sparse",
        str(root.parent / "sparse_embedding"),
        "--fluxway",
        str(root.parent / "fluxway"),
        "--owner-python",
        sys.executable,
    ]
    source_env = os.environ.copy()
    source_env["PYTHONPATH"] = str(root.parent / "sparse_embedding")
    source_env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=source_env,
    )
    assert "[PASS] sparse owner target wire" in result.stdout
    assert "[PASS] Fluxway sparse routing duplicate" in result.stdout
    assert "[PASS] Flare duplicate router owner" in result.stdout
    assert "[PASS] Flare private owner boundaries" in result.stdout

    # In dev environment with editable install, preflight must block source checkout
    assert result.returncode == 2
    assert "[BLOCKED] installed sparse owner API/native" in result.stdout
    assert "loaded from source checkout" in result.stdout

    # In isolated wheel environment, preflight must pass
    isolated_python = Path("/tmp/sparse_isolated_venv/bin/python")
    if isolated_python.is_file():
        wheel_command = list(command)
        wheel_command[-1] = str(isolated_python)
        wheel_env = os.environ.copy()
        wheel_env.pop("PYTHONPATH", None)
        wheel_env["PYTHONDONTWRITEBYTECODE"] = "1"
        wheel_result = subprocess.run(
            wheel_command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            env=wheel_env,
        )
        assert wheel_result.returncode == 0
        assert "[PASS] installed sparse owner API/native" in wheel_result.stdout
