from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from flare.data.connectors.adapters.file.arrow import read_record_batch_range


def _open_vortex_table(
    path: str,
    *,
    storage_options: Mapping[str, Any] | None = None,
) -> Any:
    import vortex.file

    options = dict(storage_options or {})
    kwargs: dict[str, Any] = {}
    for source_key, target_key in (
        ("aws_endpoint", "endpoint"),
        ("aws_access_key_id", "access_key_id"),
        ("aws_secret_access_key", "secret_access_key"),
        ("aws_session_token", "session_token"),
        ("aws_region", "region"),
    ):
        if source_key in options:
            kwargs[target_key] = options[source_key]
    if options.get("aws_virtual_hosted_style_request") is not None:
        kwargs["virtual_hosted_style_request"] = (
            str(options["aws_virtual_hosted_style_request"]).lower() == "true"
        )
    if options.get("aws_allow_http") is not None:
        kwargs["allow_http"] = str(options["aws_allow_http"]).lower() == "true"
    if path.startswith("s3://"):
        bucket, _, relative = path[5:].partition("/")
        store = vortex.file.S3Store(bucket, **kwargs)
        return vortex.file.open(relative, store=store, without_segment_cache=True)
    return vortex.file.open(path, without_segment_cache=True)


def read_vortex_range(
    path: str,
    *,
    projection: Sequence[str],
    row_start: int,
    row_count: int,
    storage_options: Mapping[str, Any] | None = None,
) -> pa.Table:
    table = _open_vortex_table(path, storage_options=storage_options)
    return read_record_batch_range(
        table.to_arrow(projection=list(projection)),
        row_start=row_start,
        row_count=row_count,
    )


def read_vortex_schema(
    path: str,
    *,
    storage_options: Mapping[str, Any] | None = None,
) -> pa.Schema:
    table = _open_vortex_table(path, storage_options=storage_options)
    return table.dtype.to_arrow_schema()


__all__ = ["read_vortex_range", "read_vortex_schema"]
