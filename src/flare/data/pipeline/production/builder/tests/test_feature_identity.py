from __future__ import annotations

import json

import pyarrow as pa
import pytest

from flare.data.pipeline.production.builder.encoding.protocol import MASK_48
from flare.data.pipeline.production.builder.expr_plan import ExprPlan
from flare.data.pipeline.production.builder.feature_processor import FeatureProcessor
from flare.data.schema.feature_catalog import FeatureOutputKind


def write_config(tmp_path, *, as_it_is: list[str]) -> tuple[str, str, str]:
    feature_tsv = tmp_path / "feature.tsv"
    model_config = tmp_path / "model_config.json"
    category_dir = tmp_path / "category"
    category_dir.mkdir()
    model_config.write_text(
        json.dumps(
            {
                "feature_column_index": {},
                "sample_weight_key": "weight",
                "sample_weight_map": {},
                "as_it_is": as_it_is,
                "shared_embedding_dict": {},
                "hash_bucket_size": {},
                "feature_cut_boundary": {},
                "category_file": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "fea_cross.tsv").write_text(
        "code,feature,class,p1,p2\n",
        encoding="utf-8",
    )
    (tmp_path / "fid_cross.tsv").write_text(
        "code,feature,class,p1,p2\n",
        encoding="utf-8",
    )
    return str(feature_tsv), str(model_config), str(category_dir)


def test_cross_definitions_are_keyed_by_code_not_display_name(tmp_path) -> None:
    path = tmp_path / "fea_cross.tsv"
    path.write_text(
        "code,feature,class,p1,p2\n"
        "55070,same-name,feaHitCount,1,2\n"
        "55079,same-name,feaHitCount,1,2\n",
        encoding="utf-8",
    )

    definitions = FeatureProcessor._load_cross_tsv(str(path), "fea")

    assert set(definitions) == {55070, 55079}
    assert definitions[55070].name == definitions[55079].name == "same-name"


def test_name_only_model_reference_rejects_distinct_feature_codes(tmp_path) -> None:
    feature_tsv, model_config, _ = write_config(
        tmp_path,
        as_it_is=["same-name"],
    )
    (tmp_path / "feature.tsv").write_text(
        "code\tfeature\tlength\ttype\tkey_by\n"
        "1\tsame-name\t1\tint64\ta\n"
        "2\tsame-name\t1\tint64\tb\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"ambiguous feature name.*codes=\[1, 2\]"):
        FeatureProcessor(
            feature_tsv,
            model_config,
            model_input_feature_names=["same-name"],
        )


def test_model_selection_rejects_legacy_no_use_feature(tmp_path) -> None:
    feature_tsv, model_config, _ = write_config(
        tmp_path,
        as_it_is=["selected"],
    )
    config = json.loads((tmp_path / "model_config.json").read_text(encoding="utf-8"))
    config["no_use_feature"] = ["selected"]
    (tmp_path / "model_config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "feature.tsv").write_text(
        "code\tfeature\tlength\ttype\tkey_by\n1\tselected\t1\tint64\ta\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selects legacy no_use_feature"):
        FeatureProcessor(
            feature_tsv,
            model_config,
            model_input_feature_names=["selected"],
        )


def test_selected_code_rejects_conflicting_feature_definitions(tmp_path) -> None:
    feature_tsv, model_config, _ = write_config(
        tmp_path,
        as_it_is=["selected"],
    )
    (tmp_path / "feature.tsv").write_text(
        "code\tfeature\tlength\ttype\tkey_by\n"
        "1\tselected\t1\tint64\ta\n"
        "1\tselected\t2\tint64\ta\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selected feature code 1 has conflicting"):
        FeatureProcessor(
            feature_tsv,
            model_config,
            model_input_feature_names=["selected"],
        )


def test_processor_catalog_is_the_schema_identity_source(tmp_path) -> None:
    feature_tsv, model_config, _ = write_config(
        tmp_path,
        as_it_is=["readable-name"],
    )
    (tmp_path / "feature.tsv").write_text(
        "code\tfeature\tlength\ttype\tkey_by\n55113\treadable-name\t1\tint64\ta\n",
        encoding="utf-8",
    )
    processor = FeatureProcessor(
        feature_tsv,
        model_config,
        model_input_feature_names=["readable-name"],
    )

    manifest = processor.feature_catalog.to_manifest()

    assert manifest["config_digest"] == processor.feature_catalog.config_digest
    assert manifest["features"]["55113"]["name"] == "readable-name"
    assert manifest["features"]["55113"]["output_column"] == "fid_55113"


def test_recipe_category_mapping_is_scoped_to_model_features(tmp_path) -> None:
    feature_tsv, model_config, _ = write_config(
        tmp_path,
        as_it_is=[],
    )
    (tmp_path / "feature.tsv").write_text(
        "code\tfeature\tlength\ttype\tkey_by\n"
        "1\tselected\t1\tint64\ta\n"
        "2\tunselected\t1\tint64\ta\n",
        encoding="utf-8",
    )
    payload = json.loads((tmp_path / "model_config.json").read_text())
    payload["category_file"] = {
        "selected": "selected.txt",
        "unselected": "missing.txt",
    }
    (tmp_path / "model_config.json").write_text(json.dumps(payload), encoding="utf-8")

    processor = FeatureProcessor(
        feature_tsv,
        model_config,
        category_values={"selected": {10: 0}},
        model_input_feature_names=["selected"],
    )

    assert processor.feature_catalog.require(1).name == "selected"


def test_category_sequence_schema_uses_dense_indices(tmp_path) -> None:
    feature_tsv, model_config, _ = write_config(
        tmp_path,
        as_it_is=[],
    )
    (tmp_path / "feature.tsv").write_text(
        "code\tfeature\tlength\ttype\tkey_by\n"
        "2687\tcontext_category_codes\t5\tint64\tprofile\n",
        encoding="utf-8",
    )
    payload = json.loads((tmp_path / "model_config.json").read_text())
    payload["category_file"] = {"context_category_codes": "context_category_codes"}
    (tmp_path / "model_config.json").write_text(json.dumps(payload), encoding="utf-8")

    processor = FeatureProcessor(
        feature_tsv,
        model_config,
        category_values={"context_category_codes": {-3: 0, 0: 1, 100101: 2}},
        model_input_feature_names=["context_category_codes"],
    )

    field = processor.build_vortex_schema().field("fid_2687")
    assert str(field.type) == "list<item: int32>"
    assert field.metadata[b"pad_value"] == b"0"


def test_as_it_is_precedes_auto_fallback(tmp_path) -> None:
    feature_tsv, model_config, _ = write_config(
        tmp_path,
        as_it_is=["raw_id"],
    )
    (tmp_path / "feature.tsv").write_text(
        "code\tfeature\tlength\ttype\tkey_by\n17\traw_id\t1\tint64\tcontext\n",
        encoding="utf-8",
    )
    payload = json.loads((tmp_path / "model_config.json").read_text())
    payload["category_file"] = {"raw_id": "auto"}
    (tmp_path / "model_config.json").write_text(json.dumps(payload), encoding="utf-8")

    processor = FeatureProcessor(
        feature_tsv,
        model_config,
        auto_category_features={"raw_id"},
        model_input_feature_names=["raw_id"],
    )

    assert processor.feature_catalog.require(17).encoder == "AsItIsEncoderExpr"


@pytest.mark.parametrize(
    ("config_key", "config_value", "expected_encoder"),
    [
        ("category_file", "raw_id.categories", "CategoryEncoderExpr"),
        ("hash_bucket_size", 64, "HashEncoderExpr"),
        ("feature_cut_boundary", [1.0, 2.0], "BucketEncoderExpr"),
        ("shared_embedding_dict", "target_id", "SharedEmbEncoderExpr"),
    ],
)
def test_explicit_encoder_precedes_as_it_is_and_auto(
    tmp_path,
    config_key: str,
    config_value: object,
    expected_encoder: str,
) -> None:
    feature_tsv, model_config, _ = write_config(
        tmp_path,
        as_it_is=["raw_id", "target_id"],
    )
    (tmp_path / "feature.tsv").write_text(
        "code\tfeature\tlength\ttype\tkey_by\n"
        "17\traw_id\t1\tint64\tcontext\n"
        "18\ttarget_id\t1\tint64\tcontext\n",
        encoding="utf-8",
    )
    payload = json.loads((tmp_path / "model_config.json").read_text())
    payload["category_file"] = {"raw_id": "auto"}
    payload[config_key] = {"raw_id": config_value}
    if config_key == "category_file":
        category_values = {"raw_id": {10: 0, 20: 1}}
    else:
        category_values = None
    (tmp_path / "model_config.json").write_text(json.dumps(payload), encoding="utf-8")

    processor = FeatureProcessor(
        feature_tsv,
        model_config,
        category_values=category_values,
        model_input_feature_names=["raw_id"],
    )

    assert processor.feature_catalog.require(17).encoder == expected_encoder


def test_dictionary_category_is_mapped_after_fid_cross(tmp_path) -> None:
    feature_tsv, model_config, _ = write_config(tmp_path, as_it_is=[])
    (tmp_path / "feature.tsv").write_text(
        "code\tfeature\tlength\ttype\tkey_by\n"
        "1\tcategory\t1\tint64\tprofile\n"
        "2\tother\t1\tint64\tprofile\n",
        encoding="utf-8",
    )
    (tmp_path / "fid_cross.tsv").write_text(
        "code,feature,class,p1,p2\n3,category-other,fidCross2,1,2\n",
        encoding="utf-8",
    )
    payload = json.loads((tmp_path / "model_config.json").read_text())
    payload["category_file"] = {"category": "category.txt", "other": "auto"}
    (tmp_path / "model_config.json").write_text(json.dumps(payload), encoding="utf-8")

    processor = FeatureProcessor(
        feature_tsv,
        model_config,
        category_values={"category": {-3: 0, 10: 1, 20: 2}},
        auto_category_features={"other"},
        model_input_feature_names=["category-other"],
    )
    assert (
        processor.feature_catalog.require(1).output_kind
        is FeatureOutputKind.DENSE_I32_INDICES
    )

    output_cols = ["fid_1", "fid_2", "fid_3"]
    batch = pa.table({"category": [10], "other": [2]})
    production_plan = ExprPlan(
        processor.build_fea_cross_ops(),
        processor.build_encode_specs(),
        processor.build_fid_cross_ops(),
        output_cols,
        post_fid_encode_specs=processor.build_post_fid_encode_specs(),
    )
    audit_plan = ExprPlan(
        processor.build_fea_cross_ops(),
        processor.build_legacy_fid_audit_encode_specs(),
        processor.build_fid_cross_ops(),
        output_cols,
    )

    production = production_plan.transform(batch)
    audit = audit_plan.transform(batch)

    assert production.column("fid_1").to_pylist() == [1]
    assert (production.column("fid_3").to_pylist()[0] & MASK_48) == (10 << 24) | 2
    assert (audit.column("fid_1").to_pylist()[0] & MASK_48) == 10
    assert (audit.column("fid_3").to_pylist()[0] & MASK_48) == (10 << 24) | 2


def test_auto_category_is_used_only_without_other_declaration(tmp_path) -> None:
    feature_tsv, model_config, _ = write_config(
        tmp_path,
        as_it_is=[],
    )
    (tmp_path / "feature.tsv").write_text(
        "code\tfeature\tlength\ttype\tkey_by\n17\traw_id\t1\tint64\tcontext\n",
        encoding="utf-8",
    )
    payload = json.loads((tmp_path / "model_config.json").read_text())
    payload["category_file"] = {"raw_id": "auto"}
    (tmp_path / "model_config.json").write_text(json.dumps(payload), encoding="utf-8")

    processor = FeatureProcessor(
        feature_tsv,
        model_config,
        auto_category_features={"raw_id"},
        model_input_feature_names=["raw_id"],
    )

    assert processor.feature_catalog.require(17).encoder == "AutoCategoryEncoderExpr"
