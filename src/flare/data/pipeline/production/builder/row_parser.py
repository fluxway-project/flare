"""Public delimited-record parser driven entirely by an explicit parser spec."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa

from flare.data.pipeline.production.processing.parser_spec import ParserSpec

from .encoding.dense import category_index_column
from .feature_processor import FeatureProcessor


class FeatureBuffers(dict):
    def __init__(self, scalar_validity: dict[str, np.ndarray]):
        super().__init__()
        self.scalar_validity = scalar_validity


class RowParser:
    """Parse tab-delimited records with a ``code\x03value`` feature payload.

    The parser has no business schema. Direct columns, metadata columns, their
    source positions, and the feature-payload position all come from
    :class:`ParserSpec`.
    """

    def __init__(
        self,
        parser_spec: ParserSpec,
        metadata_maps: Mapping[str, Mapping[str, int]],
    ) -> None:
        self._spec = parser_spec
        self._metadata_maps = {
            name: dict(values) for name, values in metadata_maps.items()
        }
        if set(self._metadata_maps) != set(parser_spec.metadata_column_index):
            raise ValueError("metadata maps must exactly match parser metadata columns")
        self._feature_meta = {
            feature.code: (
                feature.name,
                feature.length,
                feature.is_float,
            )
            for feature in parser_spec.features
        }
        self._code_to_base_meta = {
            feature.code: (
                feature.name,
                feature.length,
                feature.is_float,
                parser_spec.float_null_sentinel if feature.is_float else parser_spec.null_sentinel,
                np.float32 if feature.is_float else np.int64,
            )
            for feature in parser_spec.features
        }
        fast_parse = self._require_fast_parse_extension()
        self._category_table = None
        if parser_spec.category_mappings:
            self._category_table = fast_parse.CategoryTable(
                parser_spec.category_mappings,
                parser_spec.category_defaults,
            )
        metadata_slots = tuple(
            sorted(parser_spec.metadata_column_index.items(), key=lambda item: item[1])
        )
        self._metadata_slots = metadata_slots
        self._metadata_table = fast_parse.MetadataTable(
            metadata_slots,
            self._metadata_maps,
        )
        self._column_count = self._required_column_count()

    def _require_fast_parse_extension(self):
        import importlib

        return importlib.import_module(
            "flare.data.pipeline.production.builder._fast_parse"
        )

    def _required_column_count(self) -> int:
        indexes = tuple(self._spec.direct_column_index.values()) + tuple(
            self._spec.metadata_column_index.values()
        )
        if self._spec.feature_payload_column_index is not None:
            indexes += (self._spec.feature_payload_column_index,)
        return max(indexes, default=-1) + 1

    def parse_lines(
        self, lines: list[str]
    ) -> tuple[FeatureBuffers, dict[str, np.ndarray], int]:
        return self.parse_arrow(pa.array(lines, type=pa.string()))

    def parse_arrow(
        self, col: pa.Array
    ) -> tuple[FeatureBuffers, dict[str, np.ndarray], int]:
        if not pa.types.is_string(col.type) and not pa.types.is_large_string(col.type):
            raise TypeError(f"parser input must be a string array, got {col.type}")
        if isinstance(col, pa.ChunkedArray):
            if col.num_chunks != 1:
                col = col.combine_chunks()
            col = col.chunk(0)

        fast_parse = self._require_fast_parse_extension()
        n_rows = len(col)
        feature_buffers, meta_ints = self._allocate_buffers(n_rows)

        bufs = col.buffers()
        offsets_dtype = np.int64 if pa.types.is_large_string(col.type) else np.int32
        offsets = np.frombuffer(bufs[1], dtype=offsets_dtype)[
            col.offset : col.offset + n_rows + 1
        ]
        if offsets_dtype == np.int32:
            offsets = offsets.astype(np.int64)
        data_buf = bufs[2]
        if data_buf is None:
            data_buf = np.empty(0, dtype=np.uint8)

        direct_cols = self._spec.direct_column_index
        payload_index = self._spec.feature_payload_column_index
        if payload_index is None:
            raise ValueError("the compiled parser requires a feature payload column")
        return fast_parse.parse_arrow_direct(
            data_buf,
            offsets,
            n_rows,
            self._code_to_base_meta,
            tuple(sorted(direct_cols.items(), key=lambda item: item[1])),
            self._column_count,
            payload_index,
            self._metadata_slots,
            self._metadata_table,
            feature_buffers,
            meta_ints,
            self._category_table,
            self._spec.null_sentinel,
            self._spec.float_null_sentinel,
        )

    def parse_to_table(self, col: pa.ChunkedArray) -> pa.Table:
        if col.num_chunks != 1:
            col = col.combine_chunks()
        raw_features, metadata, batch_size = self.parse_arrow(col.chunk(0))
        return _raw_features_to_arrow(
            raw_features, metadata, self.raw_arrow_schema(), batch_size
        )

    def _allocate_buffers(
        self, batch_size: int
    ) -> tuple[FeatureBuffers, dict[str, np.ndarray]]:
        feature_buffers = FeatureBuffers({})
        schema_names = set(self._spec.raw_schema.names)
        for feature in self._spec.features:
            has_cat = feature.code in self._spec.category_mappings
            cat_name = category_index_column(feature.code) if has_cat else None
            needs_num = feature.name in schema_names
            shared_offsets = None

            if needs_num:
                dtype = np.float32 if feature.is_float else np.int64
                pad = (
                    np.float32(self._spec.float_null_sentinel)
                    if feature.is_float
                    else np.int64(self._spec.null_sentinel)
                )
                if feature.length == 1:
                    feature_buffers[feature.name] = np.full(batch_size, pad, dtype=dtype)
                    feature_buffers.scalar_validity[feature.name] = np.zeros(
                        batch_size, dtype=np.uint8
                    )
                else:
                    shared_offsets = np.zeros(batch_size + 1, dtype=np.int32)
                    feature_buffers[feature.name] = (
                        np.empty(batch_size * feature.length, dtype=dtype),
                        shared_offsets,
                    )

            if has_cat:
                default_idx = self._spec.category_defaults[feature.code]
                if feature.length == 1:
                    feature_buffers[cat_name] = np.full(
                        batch_size, default_idx, dtype=np.int32
                    )
                    feature_buffers.scalar_validity[cat_name] = np.ones(
                        batch_size, dtype=np.uint8
                    )
                else:
                    if shared_offsets is None:
                        shared_offsets = np.zeros(batch_size + 1, dtype=np.int32)
                    feature_buffers[cat_name] = (
                        np.zeros(batch_size * feature.length, dtype=np.int32),
                        shared_offsets,
                    )

        metadata = {
            name: np.zeros(batch_size, dtype=np.int32)
            for name in self._spec.metadata_column_index
        }
        for name in self._spec.direct_column_index:
            if name not in feature_buffers:
                feature_buffers[name] = np.full(
                    batch_size, self._spec.null_sentinel, dtype=np.int64
                )
                feature_buffers.scalar_validity[name] = np.zeros(
                    batch_size, dtype=np.uint8
                )
        return feature_buffers, metadata

    def raw_arrow_schema(self) -> pa.Schema:
        return self._spec.raw_schema


def _raw_features_to_arrow(
    raw_features: FeatureBuffers,
    metadata: dict[str, np.ndarray],
    schema: pa.Schema,
    batch_size: int,
) -> pa.Table:
    arrays: list[pa.Array] = []
    for field in schema:
        if field.name in raw_features:
            value = raw_features[field.name]
            if pa.types.is_list(field.type):
                flat, offsets = value
                total_elements = int(offsets[batch_size])
                arrays.append(
                    pa.ListArray.from_arrays(
                        pa.array(offsets, type=pa.int32()),
                        pa.array(flat[:total_elements], type=field.type.value_type),
                    )
                )
            else:
                validity = raw_features.scalar_validity[field.name]
                arrays.append(pa.array(value, mask=validity == 0, type=field.type))
        else:
            arrays.append(pa.array(metadata[field.name], type=field.type))
    if any(len(array) != batch_size for array in arrays):
        raise RuntimeError("parsed columns have inconsistent row counts")
    return pa.table(arrays, schema=schema)


def _load_model_input_feature_names(model_conf_dir: Path) -> list[str]:
    import importlib.util

    model_structure_path = model_conf_dir / "model_structure.py"
    spec = importlib.util.spec_from_file_location(
        "flare_model_structure", model_structure_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load model structure from {model_structure_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    feature_field = module.feature_field
    if not isinstance(feature_field, dict):
        raise TypeError("model structure feature_field must be a dict")
    required: set[str] = set()
    for names in feature_field.values():
        if not isinstance(names, Iterable) or isinstance(names, (str, bytes)):
            raise TypeError("model structure feature groups must be iterable")
        required.update(names)
    return sorted(required)


def build_feature_processor(
    conf_dir: Path,
    model_conf_dir: Path,
    metadata_columns: Mapping[str, int] | None = None,
    category_values: Mapping[str, Mapping[str, int]] | None = None,
    auto_category_features: Iterable[str] = (),
) -> FeatureProcessor:
    feature_tsv = conf_dir / "conf_common" / "feature.tsv"
    model_config = model_conf_dir / "model_config.json"
    return FeatureProcessor(
        feature_tsv=str(feature_tsv),
        model_config_json=str(model_config),
        category_feature_dir=None,
        category_values=category_values,
        auto_category_features=auto_category_features,
        model_input_feature_names=_load_model_input_feature_names(model_conf_dir),
        metadata_columns=metadata_columns,
    )
