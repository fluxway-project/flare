"""Public delimited-record parser driven entirely by an explicit parser spec."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa

from flare.data.pipeline.production.builder.constants import (
    DEFAULT_FLOAT_NULL_SENTINEL,
    DEFAULT_NULL_SENTINEL,
)
from flare.data.pipeline.production.processing.parser_spec import ParserSpec

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
                -3.0 if feature.is_float else -3,
                np.float32 if feature.is_float else np.int64,
            )
            for feature in parser_spec.features
        }
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
        offsets = np.frombuffer(bufs[1], dtype=offsets_dtype)[: n_rows + 1]
        if offsets_dtype == np.int32:
            offsets = offsets.astype(np.int64)
        data_buf = bufs[2]
        if data_buf is None:
            data_buf = np.empty(0, dtype=np.uint8)

        direct_cols = self._spec.direct_column_index
        meta_cols = self._spec.metadata_column_index
        metadata_slots = tuple(sorted(meta_cols.items(), key=lambda item: item[1]))
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
            metadata_slots,
            self._metadata_maps,
            feature_buffers,
            meta_ints,
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
        for name, length, is_float in self._feature_meta.values():
            dtype = np.float32 if is_float else np.int64
            pad = (
                np.float32(DEFAULT_FLOAT_NULL_SENTINEL)
                if is_float
                else np.int64(DEFAULT_NULL_SENTINEL)
            )
            if length == 1:
                feature_buffers[name] = np.full(batch_size, pad, dtype=dtype)
                feature_buffers.scalar_validity[name] = np.zeros(
                    batch_size, dtype=np.uint8
                )
            else:
                feature_buffers[name] = (
                    np.empty(batch_size * length, dtype=dtype),
                    np.zeros(batch_size + 1, dtype=np.int32),
                )
        metadata = {
            name: np.zeros(batch_size, dtype=np.int32)
            for name in self._spec.metadata_column_index
        }
        for name in self._spec.direct_column_index:
            if name not in feature_buffers:
                feature_buffers[name] = np.full(
                    batch_size, DEFAULT_NULL_SENTINEL, dtype=np.int64
                )
                feature_buffers.scalar_validity[name] = np.zeros(
                    batch_size, dtype=np.uint8
                )
        return feature_buffers, metadata

    def _parse_record(
        self,
        line: str,
        row_index: int,
        feature_buffers: FeatureBuffers,
        metadata: dict[str, np.ndarray],
        cursors: dict[str, int],
    ) -> None:
        columns = line.rstrip("\n").split("\t")
        if len(columns) < self._column_count:
            raise ValueError(
                f"record has {len(columns)} columns, expected at least {self._column_count}"
            )
        for name, column_index in self._spec.direct_column_index.items():
            value = columns[column_index]
            if value and value != "0":
                feature_buffers[name][row_index] = _parse_integer(value)
                feature_buffers.scalar_validity[name][row_index] = 1
        for name, column_index in self._spec.metadata_column_index.items():
            value = columns[column_index]
            try:
                metadata[name][row_index] = self._metadata_maps[name][value]
            except KeyError as exc:
                raise KeyError(f"unknown metadata value for {name}: {value!r}") from exc
        payload_index = self._spec.feature_payload_column_index
        if payload_index is None:
            return
        payload = columns[payload_index]
        if not payload:
            return
        for item in payload.split("\x02"):
            if not item or "\x03" not in item:
                continue
            code_text, raw_value = item.split("\x03", 1)
            try:
                code = int(code_text)
            except ValueError:
                continue
            feature = self._feature_meta.get(code)
            if feature is None:
                continue
            name, length, is_float = feature
            value = raw_value.strip("[]")
            if not value:
                continue
            values = value.split(",") if length > 1 else [value]
            parsed = [
                float(v) if is_float else _parse_integer(v) for v in values[:length]
            ]
            if length == 1:
                feature_buffers[name][row_index] = parsed[0]
                feature_buffers.scalar_validity[name][row_index] = 1
            else:
                flat, _offsets = feature_buffers[name]
                start = cursors[name]
                end = start + len(parsed)
                if end > flat.size:
                    raise ValueError(
                        f"feature {name!r} exceeds configured vector capacity {length}"
                    )
                flat[start:end] = parsed
                cursors[name] = end

    def raw_arrow_schema(self) -> pa.Schema:
        fields: list[pa.Field] = []
        for name, length, is_float in self._feature_meta.values():
            value_type = pa.float32() if is_float else pa.int64()
            fields.append(
                pa.field(name, value_type if length == 1 else pa.list_(value_type))
            )
        feature_names = {value[0] for value in self._feature_meta.values()}
        for name in self._spec.direct_column_index:
            if name not in feature_names:
                fields.append(pa.field(name, pa.int64()))
        for name in self._spec.metadata_column_index:
            fields.append(pa.field(name, pa.int32()))
        schema = pa.schema(fields)
        if not schema.equals(self._spec.raw_schema, check_metadata=True):
            raise RuntimeError(
                "parser runtime schema differs from compiled parser spec"
            )
        return schema


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


def _parse_integer(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        decimal = float(value)
        if not decimal.is_integer():
            raise ValueError(f"integer feature value is not integral: {value!r}")
        return int(decimal)


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
    category_values: Mapping[str, Mapping[int, int]] | None = None,
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
