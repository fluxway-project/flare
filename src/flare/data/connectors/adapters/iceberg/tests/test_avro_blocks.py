from io import BytesIO

import pytest
from pyiceberg.avro.file import AvroOutputFile
from pyiceberg.avro.resolver import resolve_reader
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import MANIFEST_LIST_FILE_SCHEMAS, ManifestContent, ManifestFile

from flare.data.connectors.adapters.iceberg.avro_blocks import BoundedAvroBlocks


def _long(value: int) -> bytes:
    encoded = (value << 1) ^ (value >> 63)
    result = bytearray()
    while encoded & ~0x7F:
        result.append((encoded & 0x7F) | 0x80)
        encoded >>= 7
    result.append(encoded)
    return bytes(result)


def _bytes(value: bytes) -> bytes:
    return _long(len(value)) + value


def _file(payload: bytes = b"abc", codec: bytes = b"null") -> bytes:
    return (
        b"Obj\x01"
        + _long(1)
        + _bytes(b"avro.codec")
        + _bytes(codec)
        + _long(0)
        + b"0" * 16
        + _long(2)
        + _long(len(payload))
        + payload
    )


def test_blocks_are_streamed_and_resumable() -> None:
    reader = BoundedAvroBlocks(BytesIO(_file()), max_block_bytes=3)
    block = next(reader)
    assert block.index == 0
    assert block.start_offset == reader.offset - len(block.payload) - 2
    assert block.record_count == 2
    assert block.payload == b"abc"
    assert block.next_offset == reader.offset

    resumed = BoundedAvroBlocks(BytesIO(_file()), max_block_bytes=3)
    resumed.seek(block.next_offset, block_index=1)
    with pytest.raises(StopIteration):
        next(resumed)


def test_block_budget_fails_before_reading_payload() -> None:
    reader = BoundedAvroBlocks(BytesIO(_file(b"abcd")), max_block_bytes=3)
    with pytest.raises(ValueError, match="byte budget"):
        next(reader)


def test_invalid_magic_fails_fast() -> None:
    with pytest.raises(ValueError, match="magic"):
        BoundedAvroBlocks(BytesIO(b"bad!"), max_block_bytes=1)


def test_current_block_decodes_without_materializing_other_blocks() -> None:
    class Reader:
        def read(self, decoder):
            return decoder.read_bytes()

    payload = _bytes(b"one") + _bytes(b"two")
    reader = BoundedAvroBlocks(BytesIO(_file(payload)), max_block_bytes=len(payload))
    block = next(reader)
    assert reader.decode_records(block, Reader()) == (b"one", b"two")
    assert tuple(reader.iter_records(block, Reader())) == (b"one", b"two")


class _ObservedReads(BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.read_ranges = []

    def read(self, size=-1):
        start = self.tell()
        result = super().read(size)
        self.read_ranges.append((start, self.tell()))
        return result


@pytest.mark.parametrize("preceding_blocks", [1, 32, 256])
def test_native_metadata_resume_skips_preceding_blocks(tmp_path, preceding_blocks):
    schema = MANIFEST_LIST_FILE_SCHEMAS[2]
    metadata_path = tmp_path / "manifest-list.avro"
    manifest = ManifestFile.from_args(
        manifest_path="retained-manifest.avro",
        manifest_length=100,
        partition_spec_id=0,
        content=ManifestContent.DATA,
        sequence_number=1,
        min_sequence_number=1,
        added_snapshot_id=7,
        added_files_count=1,
        existing_files_count=0,
        deleted_files_count=0,
        added_rows_count=8,
        existing_rows_count=0,
        deleted_rows_count=0,
        partitions=[],
    )
    with AvroOutputFile(
        PyArrowFileIO().new_output(str(metadata_path)),
        schema,
        "manifest_file",
        metadata={"avro.codec": "null"},
    ) as writer:
        for _ in range(preceding_blocks + 1):
            writer.write_block([manifest])
    payload = metadata_path.read_bytes()
    original = BoundedAvroBlocks(BytesIO(payload), max_block_bytes=len(payload))
    header_end = original.offset
    blocks = tuple(original)
    target = blocks[-1]
    stream = _ObservedReads(payload)
    restored = BoundedAvroBlocks(stream, max_block_bytes=target.stored_bytes)
    restored.seek(target.start_offset, block_index=target.index)
    resumed_block = next(restored)
    resolver = resolve_reader(restored.schema, schema, {-1: ManifestFile})
    (resumed_manifest,) = restored.decode_records(resumed_block, resolver)
    assert resumed_manifest.manifest_path == manifest.manifest_path
    assert resumed_manifest.added_rows_count == 8
    assert resumed_block == target
    assert all(
        end <= header_end or start >= target.start_offset
        for start, end in stream.read_ranges
    )
    assert sum(end - start for start, end in stream.read_ranges) == (
        header_end + target.next_offset - target.start_offset
    )


@pytest.mark.parametrize("codec", ["deflate", "bzip2", "zstandard", "snappy"])
def test_compressed_blocks_enforce_decoded_limit(codec):
    import zlib

    from pyarrow import Codec
    from pyiceberg.avro.codecs import KNOWN_CODECS

    raw = b"x" * 4096
    if codec == "snappy":
        compressed = Codec("snappy").compress(raw).to_pybytes()
        compressed += zlib.crc32(raw).to_bytes(4, "big")
    else:
        compressed, _ = KNOWN_CODECS[codec].compress(raw)
    assert len(compressed) < len(raw)
    payload = _file(compressed, codec.encode())
    reader = BoundedAvroBlocks(BytesIO(payload), max_block_bytes=len(raw))
    assert next(reader).payload == raw
    limited = BoundedAvroBlocks(BytesIO(payload), max_block_bytes=len(raw) - 1)
    with pytest.raises(ValueError, match="decoded block"):
        next(limited)


def test_header_limit_rejects_before_reading_oversized_value():
    payload = b"Obj\x01" + _long(1) + _bytes(b"avro.schema") + _bytes(b"x" * 4096)
    observed = _ObservedReads(payload)
    with pytest.raises(ValueError, match="header"):
        BoundedAvroBlocks(observed, max_block_bytes=4096, max_header_bytes=128)
    assert max(end for _, end in observed.read_ranges) < 128


def test_snappy_checksum_is_verified():
    import zlib

    from pyarrow import Codec

    raw = b"data"
    compressed = Codec("snappy").compress(raw).to_pybytes()
    payload = compressed + (zlib.crc32(raw) ^ 1).to_bytes(4, "big")
    reader = BoundedAvroBlocks(BytesIO(_file(payload, b"snappy")), max_block_bytes=128)
    with pytest.raises(ValueError, match="checksum"):
        next(reader)
