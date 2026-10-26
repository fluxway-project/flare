from copy import deepcopy

import pytest

from tools.validation.validate_reader_locality import validate


def allocation():
    nodes = {"read": ["b", "a", "b", "a"], "input": ["a", "b"]}
    return {
        "reader_spmd_id": "read",
        "input_spmd_id": "input",
        "reader_execution_groups": [[1, 2], [3, 0]],
        "allocation": {
            "status": "READY",
            "attested_node_ids": nodes,
            "assignments": [
                {"spmd_id": spmd, "rank": rank, "node_id": node}
                for spmd, members in nodes.items()
                for rank, node in enumerate(members)
            ],
        },
    }


def test_uses_compiled_lanes_with_permuted_physical_ranks():
    result = validate(allocation())
    assert result["runtime_nodes"] == 2
    assert result["lanes"] == [
        {"input_rank": 0, "reader_ranks": [1, 3], "node_id": "a"},
        {"input_rank": 1, "reader_ranks": [2, 0], "node_id": "b"},
    ]
    assert result["measured_transport_bytes"] is False


def test_rejects_misplaced_reader_and_unattested_plan():
    wrong = allocation()
    wrong["reader_execution_groups"] = [[0, 2], [3, 1]]
    with pytest.raises(ValueError, match="crosses runtime nodes"):
        validate(wrong)
    wrong = deepcopy(allocation())
    wrong["allocation"]["assignments"][0]["node_id"] = "a"
    with pytest.raises(ValueError, match="differs from runtime attestation"):
        validate(wrong)
