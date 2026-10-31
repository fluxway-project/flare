import pickle

from flare.data.contracts.publication import (
    PublicationIdentity,
    PublicationLease,
    PublicationProtocol,
)


def test_publication_value_objects_roundtrip():
    identity = PublicationIdentity("publication-1")
    lease = PublicationLease(("source-a",), PublicationProtocol.ICEBERG_SNAPSHOT)
    assert pickle.loads(pickle.dumps(identity)) == identity
    assert PublicationLease.from_state(lease.to_state()) == lease
