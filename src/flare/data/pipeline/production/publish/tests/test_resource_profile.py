"""Tests for HardwareProfiler and StageTopologyPlanner."""

from __future__ import annotations

import pytest

from flare.data.pipeline.production.publish.resource_profile import (
    HardwareProfile,
    HardwareProfiler,
    StageTopologyPlanner,
)

# ---------------------------------------------------------------------------
# HardwareProfile tests
# ---------------------------------------------------------------------------


class TestHardwareProfile:
    """Unit tests for HardwareProfile dataclass."""

    def test_has_gpu_true(self):
        profile = HardwareProfile(
            cpu_count=8,
            memory_bytes=32 * 1024**3,
            gpu_count=2,
            gpu_memory_bytes=16 * 1024**3,
            has_nvcomp=True,
        )
        assert profile.has_gpu is True
        assert profile.has_full_gpu_stack is True

    def test_has_gpu_false(self):
        profile = HardwareProfile(
            cpu_count=8,
            memory_bytes=32 * 1024**3,
            gpu_count=0,
            gpu_memory_bytes=0,
            has_nvcomp=False,
        )
        assert profile.has_gpu is False
        assert profile.has_full_gpu_stack is False

    def test_partial_gpu_stack(self):
        """GPU present but missing nvcomp."""
        profile = HardwareProfile(
            cpu_count=8,
            memory_bytes=32 * 1024**3,
            gpu_count=1,
            gpu_memory_bytes=16 * 1024**3,
            has_nvcomp=False,
        )
        assert profile.has_gpu is True
        assert profile.has_full_gpu_stack is False


# ---------------------------------------------------------------------------
# HardwareProfiler tests
# ---------------------------------------------------------------------------


class TestHardwareProfiler:
    """Unit tests for hardware detection."""

    def test_detect_returns_profile(self, monkeypatch):
        profiler = HardwareProfiler()
        monkeypatch.setattr(profiler, "_detect_cpu_count", lambda: 2)
        monkeypatch.setattr(profiler, "_detect_memory", lambda: 8 * 1024**3)
        monkeypatch.setattr(profiler, "_detect_gpu", lambda: (0, 0))
        monkeypatch.setattr(profiler, "_probe_import", lambda _name: False)

        profile = profiler.detect()

        assert isinstance(profile, HardwareProfile)
        assert profile == HardwareProfile(
            cpu_count=2,
            memory_bytes=8 * 1024**3,
            gpu_count=0,
            gpu_memory_bytes=0,
            has_nvcomp=False,
        )

    def test_cpu_count_positive(self):
        profiler = HardwareProfiler()
        count = profiler._detect_cpu_count()
        assert count >= 1

    def test_memory_positive(self):
        profiler = HardwareProfiler()
        mem = profiler._detect_memory()
        assert mem > 0

    def test_probe_import_existing_module(self):
        """Should return True for stdlib modules."""
        assert HardwareProfiler._probe_import("os") is True
        assert HardwareProfiler._probe_import("json") is True

    def test_probe_import_nonexistent_module(self):
        """Should return False for non-existent modules."""
        assert HardwareProfiler._probe_import("nonexistent_fake_module_xyz") is False


# ---------------------------------------------------------------------------
# StageTopologyPlanner tests
# ---------------------------------------------------------------------------


class TestStageTopologyPlanner:
    """Unit tests for topology planning."""

    def test_encoders_per_gpu_inference(self):
        """StageTopologyPlanner infers encoders_per_gpu from VRAM."""
        planner = StageTopologyPlanner()
        # 80 GiB -> ~6 GiB per encoder -> 13, capped at 4
        assert planner.infer_encoders_per_gpu(80 * 1024**3) == 4
        # 16 GiB -> ~6 GiB per encoder -> 2
        assert planner.infer_encoders_per_gpu(16 * 1024**3) == 2
        with pytest.raises(ValueError, match="gpu_memory_bytes"):
            planner.infer_encoders_per_gpu(0)
