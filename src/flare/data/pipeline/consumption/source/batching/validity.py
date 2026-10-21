from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchValidity:
    batch_size: int
    samples_valid_count: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.samples_valid_count < 0 or self.samples_valid_count > self.batch_size:
            raise ValueError(
                "samples_valid_count must be within batch_size: "
                f"samples_valid_count={self.samples_valid_count} batch_size={self.batch_size}"
            )

    @property
    def padded_count(self) -> int:
        return self.batch_size - self.samples_valid_count

    @property
    def is_complete(self) -> bool:
        return self.samples_valid_count == self.batch_size


__all__ = ["BatchValidity"]
