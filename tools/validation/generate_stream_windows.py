"""Create fixed-size producer windows for live Gravitino validation."""

from __future__ import annotations

import argparse
import io
import os
from contextlib import ExitStack
from pathlib import Path

import pyarrow as pa
import zstandard as zstd


def read_source_rows(
    path: Path,
    *,
    compression: str,
    required_rows: int,
) -> list[bytes]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[bytes] = []
    with ExitStack() as stack:
        raw = stack.enter_context(path.open("rb"))
        if compression == "plain":
            stream = raw
        elif compression == "zstd":
            decompressed = stack.enter_context(
                zstd.ZstdDecompressor().stream_reader(raw)
            )
            stream = stack.enter_context(io.BufferedReader(decompressed))
        else:
            raise ValueError(f"unsupported source compression: {compression!r}")
        for line in stream:
            row = line.rstrip(b"\r\n")
            if not row:
                raise ValueError(f"source contains an empty row: {path}")
            rows.append(row)
            if len(rows) == required_rows:
                break
    if len(rows) != required_rows:
        raise ValueError(f"source has {len(rows)} rows, requires {required_rows}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--rows-per-window", type=int, default=64)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument(
        "--source-compression",
        choices=("plain", "zstd"),
        default="plain",
    )
    parser.add_argument("--compression", choices=("plain", "zstd"), default="plain")
    args = parser.parse_args()
    if args.rows_per_window <= 0 or args.windows <= 0 or args.start_window < 0:
        raise ValueError(
            "rows-per-window and windows must be positive and start-window non-negative"
        )
    required = args.rows_per_window * (args.start_window + args.windows)
    rows = read_source_rows(
        args.source,
        compression=args.source_compression,
        required_rows=required,
    )
    args.target.mkdir(parents=True, exist_ok=True)
    for index in range(args.windows):
        source_window = args.start_window + index
        start = source_window * args.rows_per_window
        chunk = rows[start : start + args.rows_per_window]
        payload = b"\n".join(row + b"\t" for row in chunk) + b"\n"
        suffix = ".zst" if args.compression == "zstd" else ".txt"
        path = args.target / f"part-{source_window:05d}{suffix}"
        if path.exists():
            raise FileExistsError(path)
        temporary = args.target / f".{path.name}.{os.getpid()}.tmp"
        encoded = (
            pa.compress(payload, codec="zstd").to_pybytes()
            if args.compression == "zstd"
            else payload
        )
        try:
            temporary.write_bytes(encoded)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        if args.compression == "zstd":
            with pa.input_stream(path, compression="zstd") as stream:
                decoded = stream.read()
        else:
            decoded = path.read_bytes()
        assert len(decoded.splitlines()) == args.rows_per_window
    print(
        f"windows={args.windows} rows_per_window={args.rows_per_window} "
        f"source_compression={args.source_compression} "
        f"compression={args.compression}"
    )


if __name__ == "__main__":
    main()
