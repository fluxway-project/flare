from __future__ import annotations

from collections.abc import Iterable

from flare.data.contracts.transform import BatchTransform, TransformFactory


class TransformRegistry:
    def __init__(
        self,
        factories: Iterable[TransformFactory[BatchTransform]],
    ) -> None:
        indexed: dict[str, TransformFactory[BatchTransform]] = {}
        for factory in factories:
            transform_id = factory.transform_id
            if not transform_id:
                raise ValueError("transform factory id must not be empty")
            if transform_id in indexed:
                raise ValueError(f"duplicate transform factory: {transform_id}")
            indexed[transform_id] = factory
        if not indexed:
            raise ValueError("transform registry must not be empty")
        self._factories = indexed

    @property
    def transform_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def require(self, transform_id: str) -> TransformFactory[BatchTransform]:
        try:
            return self._factories[transform_id]
        except KeyError as exc:
            raise KeyError(f"unknown transform factory: {transform_id}") from exc
