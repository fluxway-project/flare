"""Bounded streaming reads of Avro container blocks."""

from __future__ import annotations

import bz2
import json
import zlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any, BinaryIO

from pyarrow import Codec
from pyiceberg.avro.decoder import new_decoder
from pyiceberg.utils.schema_conversion import AvroSchemaConversion

DEFAULT_AVRO_HEADER_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class AvroBlock:
    index: int
    record_count: int
    payload: bytes
    start_offset: int
    next_offset: int
    stored_bytes: int


class BoundedAvroBlocks:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        max_block_bytes: int,
        max_header_bytes: int = DEFAULT_AVRO_HEADER_BYTES,
    ) -> None:
        if not isinstance(max_block_bytes, int) or isinstance(max_block_bytes, bool):
            raise TypeError("max_block_bytes must be an integer")
        if max_block_bytes <= 0:
            raise ValueError("max_block_bytes must be positive")
        if type(max_header_bytes) is not int or max_header_bytes <= 0:
            raise ValueError("max_header_bytes must be a positive integer")
        self._max_header_bytes = max_header_bytes
        self._stream = stream
        self._max_block_bytes = max_block_bytes
        self._index = 0
        self._has_block = False
        self._metadata: dict[str, bytes] = {}
        self._sync = b""
        self._codec = "null"
        self._schema: Any = None
        self._read_header()

    @property
    def offset(self) -> int:
        return self._stream.tell()

    def seek(self, offset: int, *, block_index: int) -> None:
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("Avro stream offset must be non-negative")
        if (
            not isinstance(block_index, int)
            or isinstance(block_index, bool)
            or block_index < 0
        ):
            raise ValueError("Avro block index must be non-negative")
        self._stream.seek(offset)
        self._index = block_index
        self._has_block = block_index > 0

    @property
    def metadata(self) -> dict[str, bytes]:
        return dict(self._metadata)

    @property
    def schema(self) -> Any:
        return self._schema

    def __iter__(self) -> BoundedAvroBlocks:
        return self

    def __next__(self) -> AvroBlock:
        start_offset = self._stream.tell()
        if self._has_block:
            marker = self._stream.read(16)
            if marker == b"":
                raise StopIteration
            if marker is None or len(marker) != 16:
                raise ValueError("truncated Avro sync marker")
            if marker != self._sync:
                raise ValueError("Avro sync marker does not match the header")
        count = _read_long(self._stream)
        if count is None:
            raise StopIteration
        if count < 0:
            raise ValueError("Avro block record count must be non-negative")
        encoded_size = _read_long(self._stream)
        if encoded_size is None or encoded_size < 0:
            raise ValueError("Avro block has invalid byte length")
        if encoded_size > self._max_block_bytes:
            raise ValueError("Avro block exceeds configured byte budget")
        payload = _read_exact(self._stream, encoded_size)
        payload = _decompress(payload, self._codec, self._max_block_bytes)
        block = AvroBlock(
            self._index,
            count,
            payload,
            start_offset,
            self._stream.tell(),
            encoded_size,
        )
        self._index += 1
        self._has_block = True
        return block

    def decode_payload(self, block: AvroBlock) -> object:
        if not isinstance(block, AvroBlock):
            raise TypeError("decode_payload requires AvroBlock")
        return new_decoder(block.payload)

    def decode_records(self, block: AvroBlock, reader: Any) -> tuple[Any, ...]:
        if not isinstance(block, AvroBlock):
            raise TypeError("decode_records requires AvroBlock")
        if not hasattr(reader, "read") or not callable(reader.read):
            raise TypeError("Avro record reader must expose read(decoder)")
        return tuple(self.iter_records(block, reader))

    def iter_records(self, block: AvroBlock, reader: Any):
        if not isinstance(block, AvroBlock):
            raise TypeError("iter_records requires AvroBlock")
        if not hasattr(reader, "read") or not callable(reader.read):
            raise TypeError("Avro record reader must expose read(decoder)")
        decoder = new_decoder(block.payload)
        for _ in range(block.record_count):
            yield reader.read(decoder)
        if decoder.tell() != len(block.payload):
            raise ValueError("Avro block has unread payload bytes")

    def decode_with_schema(
        self,
        block: AvroBlock,
        *,
        file_schema: Any,
        read_schema: Any,
        read_types: dict[int, Any] | None = None,
        read_enums: dict[int, Any] | None = None,
    ) -> tuple[Any, ...]:
        if file_schema is None or read_schema is None:
            raise TypeError("Avro file and read schemas are required")
        from pyiceberg.avro.resolver import resolve_reader

        reader = resolve_reader(
            file_schema,
            read_schema,
            {} if read_types is None else read_types,
            {} if read_enums is None else read_enums,
        )
        return self.decode_records(block, reader)

    def _read_header(self) -> None:
        header_start = self._stream.tell()
        magic = _read_exact(self._stream, 4)
        if magic != b"Obj\x01":
            raise ValueError("invalid Avro container magic")
        metadata_count = _read_long(self._stream)
        if metadata_count is None:
            raise ValueError("invalid Avro metadata map")
        while metadata_count != 0:
            if metadata_count < 0:
                metadata_count = -metadata_count
                block_size = _read_long(self._stream)
                if block_size is None or block_size < 0:
                    raise ValueError("invalid Avro metadata block")
            for _ in range(metadata_count):
                key = _read_bytes(
                    self._stream,
                    self._max_header_bytes - (self._stream.tell() - header_start),
                )
                value = _read_bytes(
                    self._stream,
                    self._max_header_bytes - (self._stream.tell() - header_start),
                )
                self._metadata[key.decode("utf-8")] = value
            metadata_count = _read_long(self._stream)
            if metadata_count is None:
                raise ValueError("invalid Avro metadata map")
        if self._stream.tell() - header_start + 16 > self._max_header_bytes:
            raise ValueError("Avro header exceeds configured byte limit")
        self._sync = _read_exact(self._stream, 16)
        schema_bytes = self._metadata.get("avro.schema")
        if schema_bytes is not None:
            self._schema = AvroSchemaConversion().avro_to_iceberg(
                json.loads(schema_bytes.decode("utf-8"))
            )
        codec_name = self._metadata.get("avro.codec", b"null").decode("utf-8")
        self._codec = codec_name


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if value is None or len(value) != size:
        raise ValueError("truncated Avro stream")
    return value


def _read_bytes(stream: BinaryIO, max_bytes: int) -> bytes:
    start = stream.tell()
    size = _read_long(stream)
    if size is None or size < 0:
        raise ValueError("invalid Avro byte value")
    if size + stream.tell() - start > max_bytes:
        raise ValueError("Avro header exceeds configured byte limit")
    return _read_exact(stream, size)


def _read_long(stream: BinaryIO) -> int | None:
    first = stream.read(1)
    if first == b"":
        return None
    if first is None or len(first) != 1:
        raise ValueError("truncated Avro integer")
    raw = first[0]
    value = raw & 0x7F
    shift = 7
    while raw & 0x80:
        if shift >= 64:
            raise ValueError("Avro integer exceeds 64 bits")
        part = stream.read(1)
        if part is None or len(part) != 1:
            raise ValueError("truncated Avro integer")
        raw = part[0]
        value |= (raw & 0x7F) << shift
        shift += 7
    return (value >> 1) ^ -(value & 1)


def _decompress(payload: bytes, codec: str, limit: int) -> bytes:
    if codec == "null":
        return payload
    if codec == "deflate":
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        result = decoder.decompress(payload, limit + 1)
        complete = decoder.eof
    elif codec == "bzip2":
        decoder = bz2.BZ2Decompressor()
        result = decoder.decompress(payload, max_length=limit + 1)
        complete = decoder.eof
    elif codec == "zstandard":
        from zstandard import ZstdDecompressor

        with ZstdDecompressor().stream_reader(BytesIO(payload)) as stream:
            result = stream.read(limit + 1)
        complete = True
    elif codec == "snappy":
        return _decompress_snappy(payload, limit)
    else:
        raise ValueError(f"Unsupported Avro codec: {codec}")
    if len(result) > limit:
        raise ValueError("Avro decoded block exceeds configured byte limit")
    if not complete:
        raise ValueError("truncated compressed Avro block")
    return result


def _decompress_snappy(payload: bytes, limit: int) -> bytes:
    if len(payload) < 5:
        raise ValueError("truncated Avro Snappy block")
    size = 0
    for index, byte in enumerate(payload[:5]):
        size |= (byte & 0x7F) << (7 * index)
        if not byte & 0x80:
            break
    else:
        raise ValueError("invalid Snappy uncompressed length")
    if size > limit:
        raise ValueError("Avro decoded block exceeds configured byte limit")
    result = Codec("snappy").decompress(
        memoryview(payload)[:-4], decompressed_size=size, asbytes=True
    )
    if zlib.crc32(result) != int.from_bytes(payload[-4:], "big"):
        raise ValueError("Avro Snappy checksum mismatch")
    return result
