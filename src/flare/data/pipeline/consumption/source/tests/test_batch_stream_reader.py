import numpy as np
import pytest

from flare.data.pipeline.consumption.source.batching import MappingBatchOperations
from flare.data.pipeline.consumption.source.batching.stream_reader import (
    StreamingBatchReader,
    finite_batches,
)
from flare.data.pipeline.consumption.source.contracts.model import (
    FileLocator,
    SourceRecordRange,
)


def test_finalization_returns_mapping_with_scalar_samples_valid_count() -> None:
    reader = StreamingBatchReader(
        batch_size=4,
        completion_policy="pad",
        operations=MappingBatchOperations(),
        validity_field="samples_valid_count",
    )
    list(reader.append_loader(iter([{"group_id": np.array([1, 2])}])))

    finalized = reader.finalize()

    assert len(finalized) == 1
    assert finalized[0]["group_id"].tolist() == [1, 2, 2, 2]
    assert finalized[0]["samples_valid_count"].tolist() == [2]
    assert "valid_mask" not in finalized[0]


def test_partial_tail_checkpoint_exports_ranges_and_replays_without_payload() -> None:
    reader = StreamingBatchReader(
        batch_size=4,
        completion_policy="carry",
        operations=MappingBatchOperations(),
        validity_field="samples_valid_count",
    )
    source_range = SourceRecordRange(
        "iceberg", "r1", FileLocator("part-0.parquet"), 0, 2, "schema", "parquet"
    )
    list(
        reader.append_loader(
            iter(
                [
                    (
                        {"group_id": np.array([1, 2])},
                        (source_range,),
                    )
                ]
            )
        )
    )

    state = reader.checkpoint_state()
    assert state["pending_rows"] == 2
    assert state["pending_ranges"][0]["record_start"] == 0
    assert "group_id" not in state

    restored = StreamingBatchReader(
        batch_size=4,
        completion_policy="carry",
        operations=MappingBatchOperations(),
        validity_field="samples_valid_count",
    )
    restored.restore_state(state)
    assert restored.pending_rows == 0
    assert restored.replay_ranges[0].locator == "part-0.parquet"


def test_finite_batches_applies_terminal_completion_policy_once() -> None:
    source = iter([{"value": np.asarray([1, 2, 3])}])

    dropped = list(
        finite_batches(
            source,
            batch_size=2,
            completion_policy="drop",
            operations=MappingBatchOperations(),
            validity_field="samples_valid_count",
        )
    )
    assert [item["value"].tolist() for item in dropped] == [[1, 2]]
    assert dropped[0]["samples_valid_count"].tolist() == [2]

    padded = list(
        finite_batches(
            iter([{"value": np.asarray([1, 2, 3])}]),
            batch_size=2,
            completion_policy="pad",
            operations=MappingBatchOperations(),
            validity_field="samples_valid_count",
        )
    )
    assert [item["value"].tolist() for item in padded] == [[1, 2], [3, 3]]
    assert [item["samples_valid_count"].tolist() for item in padded] == [[2], [1]]

    with pytest.raises(ValueError, match="incomplete batch"):
        list(
            finite_batches(
                iter([{"value": np.asarray([1, 2, 3])}]),
                batch_size=2,
                completion_policy="require",
                operations=MappingBatchOperations(),
                validity_field="samples_valid_count",
            )
        )


def test_finite_batches_can_leave_validity_to_consumer_neutral_reader() -> None:
    batches = list(
        finite_batches(
            iter([{"value": np.asarray([1, 2, 3])}]),
            batch_size=2,
            completion_policy="pad",
            operations=MappingBatchOperations(),
            validity_field=None,
        )
    )

    assert all("samples_valid_count" not in item for item in batches)


def test_finite_batches_with_pyarrow_table() -> None:
    import pyarrow as pa

    from flare.data.pipeline.consumption.source.batching import PyArrowTableOperations

    t1 = pa.table({"v": [1, 2, 3]})
    t2 = pa.table({"v": [4, 5]})

    batches = list(
        finite_batches(
            iter([t1, t2]),
            batch_size=2,
            completion_policy="drop",
            operations=PyArrowTableOperations(),
            validity_field="samples_valid_count",
        )
    )

    assert len(batches) == 2
    assert isinstance(batches[0], pa.Table)
    assert batches[0]["v"].to_pylist() == [1, 2]
    assert batches[0].schema.metadata[b"samples_valid_count"] == b"2"
    assert batches[1]["v"].to_pylist() == [3, 4]
    assert batches[1].schema.metadata[b"samples_valid_count"] == b"2"
