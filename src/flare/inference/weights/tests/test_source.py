from __future__ import annotations

from types import SimpleNamespace

from flare.inference.weights import source


def test_published_asset_evidence_ignores_mount_device_number(tmp_path, monkeypatch):
    asset = tmp_path / "steps" / "12"
    (asset / "model").mkdir(parents=True)
    (asset / "sparse").mkdir()
    evidence = source.PublishedAssetEvidence.capture(asset, asset, None)

    original_stat = source.Path.stat

    def stat_with_different_device(path, *args, **kwargs):
        stat = original_stat(path, *args, **kwargs)
        return SimpleNamespace(
            st_dev=stat.st_dev + 1,
            st_ino=stat.st_ino,
            st_mtime_ns=stat.st_mtime_ns,
            st_size=stat.st_size,
            st_mode=stat.st_mode,
        )

    monkeypatch.setattr(source.Path, "stat", stat_with_different_device)
    evidence.verify()
