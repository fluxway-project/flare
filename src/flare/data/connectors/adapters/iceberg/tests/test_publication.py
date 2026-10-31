from types import SimpleNamespace

import pytest

from flare.data.connectors.adapters.iceberg.publication import (
    index_publication_snapshots,
    snapshot_added_records,
)


def test_index_ignores_unlabelled_history_only_when_explicitly_requested():
    foreign = SimpleNamespace(summary={"operation": "append"})
    own = SimpleNamespace(summary={"flare.publication-id": "one"})
    with pytest.raises(RuntimeError, match="missing publication id"):
        index_publication_snapshots([foreign, own])
    assert index_publication_snapshots(
        [foreign, own], require_publication_ids=False
    ) == {"one": own}


def test_index_rejects_duplicates_before_any_caller_can_use_the_result():
    snapshots = [
        SimpleNamespace(summary={"flare.publication-id": "one"}) for _ in range(2)
    ]
    with pytest.raises(RuntimeError, match="duplicate publication ids: one"):
        index_publication_snapshots(snapshots, require_publication_ids=False)


@pytest.mark.parametrize(
    ("summary", "error"),
    [
        (None, RuntimeError),
        ([], TypeError),
        ({"flare.publication-id": ""}, RuntimeError),
    ],
)
def test_index_rejects_malformed_snapshot_metadata(summary, error):
    with pytest.raises(error):
        index_publication_snapshots([SimpleNamespace(summary=summary)])


@pytest.mark.parametrize("value", [True, -1, "-1", None, 1.5])
def test_record_counts_reject_invalid_values_consistently(value):
    with pytest.raises((ValueError, TypeError)):
        snapshot_added_records(SimpleNamespace(summary={"added-records": value}))


def test_record_counts_accept_integer_and_decimal_string_values():
    for value in [7, "7"]:
        assert (
            snapshot_added_records(SimpleNamespace(summary={"added-records": value}))
            == 7
        )
