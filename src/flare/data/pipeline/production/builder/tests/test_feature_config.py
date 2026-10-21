from pathlib import Path

import pytest

from flare.data.pipeline.production.builder.feature_processor import FeatureProcessor


def test_cross_config_preserves_duplicate_names_with_distinct_codes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cross.tsv"
    path.write_text(
        "code,name,class,left,right\n"
        "1,left-right,fidCross2,11,12\n"
        "2,left-right,fidCross2,11,12\n",
        encoding="utf-8",
    )
    definitions = FeatureProcessor._load_cross_tsv(str(path), "fid")
    assert set(definitions) == {1, 2}
    assert definitions[1].name == definitions[2].name == "left-right"


def test_cross_config_rejects_duplicate_feature_code(tmp_path: Path) -> None:
    path = tmp_path / "cross.tsv"
    path.write_text(
        "code,name,class,left,right\n"
        "1,left-right,fidCross2,11,12\n"
        "1,left-other,fidCross2,11,13\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting cross feature code"):
        FeatureProcessor._load_cross_tsv(str(path), "fid")
