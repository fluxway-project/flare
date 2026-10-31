import json

import pytest

from flare.io.artifacts import write_json


def test_write_json_replaces_complete_document(tmp_path):
    target = tmp_path / "nested" / "state.json"
    write_json(target, {"processed": ["first"]})
    write_json(target, {"processed": ["first", "second"]})
    assert json.loads(target.read_text()) == {"processed": ["first", "second"]}
    assert list(target.parent.iterdir()) == [target]


def test_exclusive_json_write_preserves_existing_report(tmp_path):
    target = tmp_path / "report.json"
    write_json(target, {"run": "original"}, exclusive=True)
    with pytest.raises(FileExistsError):
        write_json(target, {"run": "replacement"}, exclusive=True)
    assert json.loads(target.read_text()) == {"run": "original"}
    assert list(tmp_path.iterdir()) == [target]


def test_json_encoding_failure_preserves_previous_state_and_removes_temporary_file(
    tmp_path,
):
    target = tmp_path / "state.json"
    write_json(target, {"processed": ["first"]})
    with pytest.raises(TypeError):
        write_json(target, {"invalid": object()})
    assert json.loads(target.read_text()) == {"processed": ["first"]}
    assert list(tmp_path.iterdir()) == [target]
