"""Reserved device-transfer seam; GPU support is intentionally not implemented."""

from __future__ import annotations

from typing import Any


class DeviceTransferStage:
    def __init__(self, target_device: str) -> None:
        if (
            not isinstance(target_device, str)
            or not target_device
            or target_device == "cpu"
        ):
            raise ValueError("DeviceTransferStage requires an explicit non-CPU device")
        self.target_device = target_device

    def transfer(self, value: Any) -> Any:
        raise NotImplementedError(
            "GPU/device transfer is not implemented in this release"
        )
