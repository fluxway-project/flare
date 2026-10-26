from collections.abc import Iterable, Mapping, Sequence


def _positive_worker_count(value: str, field: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an integer, got {value!r}") from error
    if count <= 0:
        raise ValueError(f"{field} must be positive, got {count}")
    return count


def parse_sparse_group_name(group_name: str) -> tuple[str, int]:
    if not isinstance(group_name, str) or not group_name:
        raise ValueError(
            f"sparse group name must be a non-empty string, got {group_name!r}"
        )
    if "_dim_" not in group_name:
        raise ValueError(
            f"sparse group name must contain '_dim_<dimension>', got {group_name!r}"
        )
    base_name, dimension_text = group_name.rsplit("_dim_", 1)
    if not base_name:
        raise ValueError(f"sparse group base name must not be empty in {group_name!r}")
    try:
        dimension = int(dimension_text)
    except ValueError as error:
        raise ValueError(
            f"invalid dimension in sparse group name {group_name!r}: {dimension_text!r}"
        ) from error
    if dimension <= 0:
        raise ValueError(
            f"dimension in sparse group name {group_name!r} must be positive, got {dimension}"
        )
    return base_name, dimension


def extract_sparse_groups_and_dimensions(
    sparse_id_groups: Mapping[str, Sequence[str]] | Sequence[str],
) -> tuple[tuple[str, ...], dict[str, list[int]]]:
    if isinstance(sparse_id_groups, (Mapping, tuple, list)):
        names = tuple(sparse_id_groups)
    else:
        raise TypeError(
            "sparse_id_groups must be a Mapping or Sequence of group names, "
            f"got {type(sparse_id_groups).__name__}"
        )

    dimensions_by_base: dict[str, set[int]] = {}
    for group_name in names:
        base_name, dimension = parse_sparse_group_name(group_name)
        dimensions_by_base.setdefault(base_name, set()).add(dimension)

    sorted_dimensions = {
        base_name: sorted(dims) for base_name, dims in dimensions_by_base.items()
    }
    return names, sorted_dimensions


def extract_sparse_base_names(sparse_groups: Iterable[str]) -> tuple[str, ...]:
    if not isinstance(sparse_groups, Iterable):
        raise TypeError("sparse_groups must be an Iterable")
    base_names = set()
    for name in sparse_groups:
        base_name, _ = parse_sparse_group_name(name)
        base_names.add(base_name)
    return tuple(sorted(base_names))


def parse_sparse_workers(
    arg_str: str,
    sparse_groups: Iterable[str],
) -> dict[str, int]:
    base_names = extract_sparse_base_names(sparse_groups)
    config = dict.fromkeys(base_names, 1)
    if not arg_str:
        return config
    if ":" not in arg_str:
        worker_count = _positive_worker_count(arg_str, "sparse worker count")
        return dict.fromkeys(base_names, worker_count)
    configured = set()
    for item in arg_str.split(","):
        fields = item.split(":")
        if len(fields) != 2:
            raise ValueError(
                f"sparse worker override must use name:count, got {item!r}"
            )
        name, value = (field.strip() for field in fields)
        if name not in config:
            raise ValueError(f"unknown sparse group {name!r}")
        if name in configured:
            raise ValueError(f"duplicate sparse group {name!r}")
        config[name] = _positive_worker_count(
            value, f"sparse worker count for {name!r}"
        )
        configured.add(name)
    return config


def build_sparse_group_feature_widths(
    categorical_feature_groups: Mapping[str, Sequence[str]],
    feature_lengths: Mapping[str, int],
    sparse_groups: Sequence[str],
) -> dict[str, tuple[int, int]]:
    if not isinstance(categorical_feature_groups, Mapping):
        raise TypeError("categorical_feature_groups must be a Mapping")
    widths: dict[str, tuple[int, int]] = {}
    for group in sparse_groups:
        if group not in categorical_feature_groups:
            raise KeyError(
                f"sparse group {group!r} is absent from model_builder.pack_meta"
            )
        features = tuple(categorical_feature_groups[group])
        if not features:
            raise ValueError(
                f"sparse group {group!r} must contain at least one feature"
            )
        missing = [feature for feature in features if feature not in feature_lengths]
        if missing:
            raise KeyError(
                f"sparse group {group!r} has unknown feature lengths: {missing}"
            )
        lengths = tuple(feature_lengths[feature] for feature in features)
        if any(length <= 0 for length in lengths):
            raise ValueError(
                f"sparse group {group!r} has invalid feature lengths: {lengths}"
            )
        widths[group] = (len(features), max(lengths))
    return widths


__all__ = [
    "build_sparse_group_feature_widths",
    "extract_sparse_base_names",
    "extract_sparse_groups_and_dimensions",
    "parse_sparse_group_name",
    "parse_sparse_workers",
]
