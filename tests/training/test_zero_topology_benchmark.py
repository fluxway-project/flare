from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_node_local_zero_matches_global_dp_on_asymmetric_cpu_inputs() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.validation.benchmark_zero_topology import self_test; self_test()",
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ZERO_TOPOLOGY_SELF_TEST=PASS" in result.stdout
