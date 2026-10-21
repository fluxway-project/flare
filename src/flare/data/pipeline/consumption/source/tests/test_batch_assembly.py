import numpy as np
import pytest

from flare.data.pipeline.consumption.source.batching import (
    COMPLETION_POLICY_NAMES,
    BatchAssembler,
    CarryUntilComplete,
    DropIncompleteAtEnd,
    MappingBatchOperations,
    PadWithValidity,
    RequireComplete,
)
from flare.data.pipeline.consumption.source.contracts.model import (
    FileLocator,
    SourceRecordRange,
)


def _assembler(batch_size: int, policy) -> BatchAssembler:
    return BatchAssembler(batch_size, MappingBatchOperations(), policy)


def test_completion_policy_names_form_a_closed_set() -> None:
    assert COMPLETION_POLICY_NAMES == ("carry", "drop", "pad", "require")


def test_carry_policy_joins_fragments_without_dropping_boundary_tails() -> None:
    assembler = _assembler(4, CarryUntilComplete())

    first = assembler.append({"labels": np.asarray([0, 1, 2])})
    second = assembler.append({"labels": np.asarray([3, 4])})

    assert first == ()
    assert len(second) == 1
    assert second[0].payload["labels"].tolist() == [0, 1, 2, 3]
    assert assembler.pending_rows == 1
    assert assembler.finish_boundary(terminal=False) == ()
    assert assembler.pending_rows == 1


def test_drop_policy_discards_only_explicit_terminal_tail() -> None:
    assembler = _assembler(4, DropIncompleteAtEnd())
    assembler.append({"labels": np.asarray([0, 1])})

    assert assembler.finish_boundary(terminal=False) == ()
    assert assembler.pending_rows == 2
    assert assembler.finish_boundary(terminal=True) == ()
    assert assembler.pending_rows == 0


def test_pad_policy_returns_validity_without_changing_static_shape() -> None:
    assembler = _assembler(4, PadWithValidity())
    assembler.append({"labels": np.asarray([7, 8])})

    output = assembler.finish_boundary(terminal=True)

    assert len(output) == 1
    assert output[0].payload["labels"].tolist() == [7, 8, 8, 8]
    assert output[0].validity.samples_valid_count == 2
    assert output[0].validity.padded_count == 2


def test_require_complete_raises_on_terminal_tail() -> None:
    assembler = _assembler(4, RequireComplete())
    assembler.append({"labels": np.asarray([1])})

    with pytest.raises(ValueError, match="incomplete batch"):
        assembler.finish_boundary(terminal=True)


def test_mapping_operations_reject_row_misalignment() -> None:
    assembler = _assembler(4, CarryUntilComplete())

    with pytest.raises(ValueError, match="row-aligned"):
        assembler.append(
            {
                "labels": np.asarray([1, 2]),
                "group_id": np.asarray([10]),
            }
        )


def test_checkpoint_restore_requires_explicit_record_format() -> None:
    assembler = _assembler(4, CarryUntilComplete())
    assembler.append(
        {"labels": np.asarray([1])},
        (
            SourceRecordRange(
                "source", "revision", FileLocator("file"), 0, 1, "schema", "parquet"
            ),
        ),
    )
    state = assembler.checkpoint_state()
    del state["pending_ranges"][0]["record_format"]

    with pytest.raises(ValueError, match="fields do not match"):
        _assembler(4, CarryUntilComplete()).restore_checkpoint_state(state)
