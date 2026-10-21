from flare.data.pipeline.consumption.source.batching.validity import BatchValidity
from flare.data.sources.range import SourceRecordRange


def test_source_value_objects_have_canonical_owners() -> None:
    assert SourceRecordRange.__module__ == "flare.data.sources.range"
    assert BatchValidity.__module__ == (
        "flare.data.pipeline.consumption.source.batching.validity"
    )
