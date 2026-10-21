from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from runtime.graph import parse_sparse_workers
from util.conf_loader import load_conf

STAGES = (
    "input_unique",
    "sparse_lookup",
    "dense_gather",
    "dense_grad",
    "sparse_update",
    "sparse_commit",
    "dense_update",
)
SPARSE_NUMERICAL_RTOL = 2e-5
SPARSE_NUMERICAL_ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--sparse-workers", required=True)
    parser.add_argument("--dense-world-size", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--sparse-max-staleness-steps", type=int, required=True)
    parser.add_argument("--conf-version", required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_records(root: Path) -> dict[str, list[dict]]:
    records = {}
    for stage in STAGES:
        paths = sorted(root.glob(f"{stage}.*.jsonl"))
        if not paths:
            raise AssertionError(f"missing trace shards for stage: {stage}")
        rows = [
            json.loads(line) for path in paths for line in path.read_text().splitlines()
        ]
        if not rows or any(row["stage"] != stage for row in rows):
            raise AssertionError(f"invalid trace records for stage: {stage}")
        records[stage] = rows
    return records


def load_model_contract(conf_version: str):
    conf = load_conf(conf_version)
    module = importlib.import_module(f"model.geek.predict.{conf_version}")
    model_builder = module.ModelBuilder(conf)
    from runtime.nodes.dense_spmd import build_sparse_group_feature_layouts

    sparse_groups = tuple(model_builder.pack_meta.sparse_id_groups)
    feature_layouts = build_sparse_group_feature_layouts(
        model_builder,
        sparse_groups,
        project_conf_root=Path(conf.project_path) / "conf",
        conf_path=Path(conf.conf_path),
    )
    model_config = json.loads((Path(conf.conf_path) / "model_config.json").read_text())
    return feature_layouts, tuple(model_config["sample_weight_map"])


def array(payload: dict, values_key: str, summary_key: str) -> np.ndarray:
    shape = tuple(int(value) for value in payload[summary_key]["shape"])
    return np.asarray(payload[values_key]).reshape(shape)


def int64_values(values) -> np.ndarray:
    minimum = -(1 << 63)
    maximum = (1 << 63) - 1
    normalized = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("sparse trace values must be signed int64 integers")
        if value < minimum or value > maximum:
            raise ValueError("sparse trace value is outside the signed int64 domain")
        normalized.append(value)
    return np.asarray(normalized, dtype=np.int64)


def add_unique(mapping: dict, key: tuple, value, context: str) -> None:
    if key in mapping:
        raise AssertionError(f"duplicate {context}: {key}")
    mapping[key] = value


def base_group(group: str) -> str:
    return group.rsplit("_dim_", 1)[0]


def allowed_sparse_versions(versions, *, step: int, max_staleness_steps: int):
    if not versions:
        raise ValueError("sparse version history must not be empty")
    oldest_allowed_step = step - max_staleness_steps - 1
    allowed = [
        value for commit_step, value in versions if commit_step >= oldest_allowed_step
    ]
    return allowed or [versions[-1][1]]


def embedding_version_staleness_steps(
    *,
    max_async_lag_steps: int,
    grad_accum_steps: int,
) -> int:
    if max_async_lag_steps < 0:
        raise ValueError("max_async_lag_steps must be non-negative")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    return max_async_lag_steps + grad_accum_steps - 1


def audit_input(
    records,
    feature_layouts,
    dense_world_size,
    expected_steps,
):
    expected_coverage = {
        (group, step, rank)
        for group in feature_layouts
        for step in range(expected_steps)
        for rank in range(dense_world_size)
    }
    actual_coverage = {
        (row["group"], int(row["step"]), int(row["rank"])) for row in records
    }
    if actual_coverage != expected_coverage:
        raise AssertionError(
            f"input coverage mismatch missing={sorted(expected_coverage - actual_coverage)[:8]} "
            f"extra={sorted(actual_coverage - expected_coverage)[:8]}"
        )

    origin_to_key = {}
    feature_counts = defaultdict(int)
    feature_step_counts = defaultdict(lambda: defaultdict(int))
    occurrence_count = 0
    for record in records:
        group = record["group"]
        feature_layout = feature_layouts[group]
        payload = record["payload"]
        rows = int(payload["rows"])
        original = int64_values(payload["original_values"])
        unique = int64_values(payload["unique_values"])
        inverse = array(payload, "inv_values", "inv").astype(np.int64)
        origins = np.asarray(payload["unique_global_positions"], dtype=np.int64)
        nested_splits = [
            np.asarray(values, dtype=np.int64)
            for values in payload.get("nested_splits_values", [])
        ]
        if len(nested_splits) != 2:
            raise AssertionError(
                f"FID trace requires row -> feature -> ID splits group={group}, "
                f"got {len(nested_splits)} levels"
            )
        row_splits, feature_splits = nested_splits
        expected_row_splits = np.arange(
            0,
            (rows + 1) * len(feature_layout),
            len(feature_layout),
            dtype=np.int64,
        )
        np.testing.assert_array_equal(row_splits, expected_row_splits)
        if feature_splits.shape != (rows * len(feature_layout) + 1,):
            raise AssertionError(
                f"FID feature splits shape mismatch group={group} "
                f"shape={feature_splits.shape}"
            )
        if feature_splits[0] != 0 or feature_splits[-1] != original.size:
            raise AssertionError(
                f"FID feature splits do not cover values group={group} "
                f"bounds={(feature_splits[0], feature_splits[-1])} "
                f"values={original.size}"
            )
        np.testing.assert_array_equal(unique[inverse], original)
        if np.unique(unique).size != unique.size:
            raise AssertionError(
                f"duplicate compact IDs group={group} step={record['step']}"
            )
        for feature_index, (feature, max_length) in enumerate(feature_layout):
            for row in range(rows):
                node = int(row_splits[row]) + feature_index
                actual_length = int(feature_splits[node + 1] - feature_splits[node])
                if actual_length < 0 or actual_length > max_length:
                    raise AssertionError(
                        f"FID feature length out of range group={group} "
                        f"feature={feature} row={row} length={actual_length} "
                        f"max_length={max_length}"
                    )
                feature_counts[feature] += actual_length
                feature_step_counts[feature][int(record["step"])] += actual_length
        occurrence_count += int(original.size)
        for origin, key in zip(origins, unique, strict=True):
            add_unique(
                origin_to_key,
                (group, int(record["step"]), int(origin)),
                int(key),
                "input origin",
            )
    missing_features = [
        feature
        for layout in feature_layouts.values()
        for feature, _ in layout
        if feature_counts[feature] <= 0
    ]
    if missing_features:
        raise AssertionError(f"configured FIDs absent from input: {missing_features}")
    for feature, counts_by_step in feature_step_counts.items():
        if set(counts_by_step) != set(range(expected_steps)):
            raise AssertionError(
                f"FID step coverage mismatch feature={feature} "
                f"steps={sorted(counts_by_step)}"
            )
    return origin_to_key, dict(sorted(feature_counts.items())), occurrence_count


def audit_lookup(records, workers, input_rows):
    lookup_rows = {}
    rank_coverage = defaultdict(set)
    for record in records:
        group = record["group"]
        world_size = workers[base_group(group)]
        step = int(record["step"])
        rank = int(record["rank"])
        rank_coverage[(group, step)].add(rank)
        payload = record["payload"]
        keys = int64_values(payload["keys_values"])
        origins = np.asarray(payload["origin_values"], dtype=np.int64)
        embeddings = array(payload, "embedding_values", "emb").astype(np.float32)
        if not (keys.size == origins.size == embeddings.shape[0]):
            raise AssertionError(
                f"lookup row mismatch group={group} step={step} rank={rank}"
            )
        for key, origin, embedding in zip(keys, origins, embeddings, strict=True):
            if int(key) % world_size != rank:
                raise AssertionError(
                    f"hash route mismatch group={group} key={key} "
                    f"expected={int(key) % world_size} actual={rank}"
                )  # pyrefly: ignore[no-matching-overload]
            row_key = (group, step, int(origin))
            if input_rows.get(row_key) != int(key):
                raise AssertionError(f"input -> lookup key mismatch: {row_key}")
            add_unique(
                lookup_rows,
                row_key,
                (int(key), embedding.copy(), rank),
                "lookup origin",
            )
    if set(lookup_rows) != set(input_rows):
        raise AssertionError("input -> lookup coverage mismatch")
    for (group, step), ranks in rank_coverage.items():
        expected = set(range(workers[base_group(group)]))
        if ranks != expected:
            raise AssertionError(
                f"Sparse rank coverage mismatch group={group} step={step}: {ranks}"
            )
    return lookup_rows, rank_coverage


def audit_dense_forward(records, lookup_rows, dense_world_size):
    gather_rows = {}
    rank_coverage = defaultdict(set)
    for record in records:
        group = record["group"]
        step = int(record["step"])
        rank = int(record["rank"])
        rank_coverage[(group, step)].add(rank)
        payload = record["payload"]
        origins = np.asarray(payload["emb_origin_values"], dtype=np.int64)
        embeddings = array(payload, "embedding_values", "emb").astype(np.float32)
        inverse = array(payload, "inv_values", "inv").astype(np.int64)
        sparse_input = array(payload, "sparse_input_values", "sparse_input").astype(
            np.float32
        )
        for origin, embedding in zip(origins, embeddings, strict=True):
            row_key = (group, step, int(origin))
            expected = lookup_rows[row_key][1]
            np.testing.assert_array_equal(embedding, expected)
            add_unique(gather_rows, row_key, embedding.copy(), "Dense gather origin")
        padded = np.concatenate(
            [embeddings, np.zeros((1, embeddings.shape[1]), dtype=np.float32)], axis=0
        )
        if np.any(inverse < 0) or np.any(inverse > embeddings.shape[0]):
            raise AssertionError(f"Dense inverse out of range: {record}")
        if inverse.ndim != 3:
            raise AssertionError(
                f"Dense inverse must have (batch, features, ids) shape, got "
                f"{inverse.shape}"
            )
        expected_sparse_input = padded[inverse].sum(axis=2)
        np.testing.assert_allclose(
            sparse_input,
            expected_sparse_input,
            rtol=2e-6,
            atol=2e-7,
        )
    if set(gather_rows) != set(lookup_rows):
        raise AssertionError("lookup -> Dense forward coverage mismatch")
    expected_ranks = set(range(dense_world_size))
    if any(ranks != expected_ranks for ranks in rank_coverage.values()):
        raise AssertionError(f"Dense rank coverage mismatch: {dict(rank_coverage)}")
    return gather_rows


def audit_gradient_and_sparse_update(
    dense_records,
    update_records,
    commit_records,
    lookup_rows,
    grad_accum_steps,
    sparse_max_staleness_steps,
):
    embedding_staleness_steps = embedding_version_staleness_steps(
        max_async_lag_steps=sparse_max_staleness_steps,
        grad_accum_steps=grad_accum_steps,
    )
    dense_rows = {}
    for record in dense_records:
        payload = record["payload"]
        origins = np.asarray(payload["grad_origin_values"], dtype=np.int64)
        gradients = array(payload, "gradient_values", "grad").astype(np.float32)
        for origin, gradient in zip(origins, gradients, strict=True):
            add_unique(
                dense_rows,
                (record["group"], int(record["step"]), int(origin)),
                gradient.copy(),
                "Dense gradient origin",
            )

    update_rows = {}
    entries_by_step = defaultdict(list)
    for record in update_records:
        group = record["group"]
        step = int(record["step"])
        rank = int(record["rank"])
        payload = record["payload"]
        keys = int64_values(payload["keys_values"])
        origins = np.asarray(payload["origin_values"], dtype=np.int64)
        gradients = array(payload, "gradient_values", "grads").astype(np.float32)
        after = array(payload, "embedding_values_after", "emb_after").astype(np.float32)
        optimizer_applied = bool(payload["optimizer_applied"])
        expected_applied = (step + 1) % grad_accum_steps == 0
        if optimizer_applied != expected_applied:
            raise AssertionError(
                f"Sparse optimizer boundary mismatch group={group} step={step}"
            )
        for key, origin, gradient, value_after in zip(
            keys, origins, gradients, after, strict=True
        ):
            row_key = (group, step, int(origin))
            lookup = lookup_rows[row_key]
            if lookup[0] != int(key) or lookup[2] != rank:
                raise AssertionError(
                    f"gradient returned to wrong Sparse worker: {row_key}"
                )
            np.testing.assert_array_equal(gradient, dense_rows[row_key])
            add_unique(update_rows, row_key, gradient.copy(), "Sparse update origin")
            entries_by_step[step].append(
                (
                    (group, rank, int(key)),
                    lookup[1],
                    gradient,
                    value_after,
                )
            )

    commits_by_step = defaultdict(dict)
    for record in commit_records:
        group = record["group"]
        step = int(record["step"])
        rank = int(record["rank"])
        payload = record["payload"]
        keys = int64_values(payload["keys_values"])
        gradients = array(payload, "gradient_values", "grads").astype(np.float32)
        after = array(payload, "embedding_values_after", "emb_after").astype(np.float32)
        if not (keys.size == gradients.shape[0] == after.shape[0]):
            raise AssertionError(
                f"Sparse commit row mismatch group={group} step={step} rank={rank}"
            )
        for key, gradient, value_after in zip(keys, gradients, after, strict=True):
            add_unique(
                commits_by_step[step],
                (group, rank, int(key)),
                (gradient.copy(), value_after.copy()),
                "Sparse commit key",
            )

    accumulated = {}
    adam_state = {}
    current_embeddings = {}
    committed_versions = defaultdict(list)
    update_checks = 0
    stale_lookup_checks = 0
    for step in sorted(entries_by_step):
        for physical_key, lookup_embedding, gradient, _ in entries_by_step[step]:
            if physical_key in current_embeddings:
                allowed_versions = allowed_sparse_versions(
                    committed_versions[physical_key],
                    step=step,
                    max_staleness_steps=embedding_staleness_steps,
                )
                if not any(
                    np.allclose(
                        lookup_embedding,
                        version,
                        rtol=SPARSE_NUMERICAL_RTOL,
                        atol=SPARSE_NUMERICAL_ATOL,
                    )
                    for version in allowed_versions
                ):
                    raise AssertionError(
                        f"Sparse lookup exceeds staleness bound step={step} "
                        f"physical_key={physical_key} "
                        f"allowed_version_count={len(allowed_versions)}"
                    )
                if not np.allclose(
                    lookup_embedding,
                    current_embeddings[physical_key],
                    rtol=SPARSE_NUMERICAL_RTOL,
                    atol=SPARSE_NUMERICAL_ATOL,
                ):
                    stale_lookup_checks += 1
            else:
                current_embeddings[physical_key] = lookup_embedding.copy()
                committed_versions[physical_key].append((step, lookup_embedding.copy()))
            accumulated[physical_key] = (
                accumulated.get(physical_key, np.zeros_like(gradient)) + gradient
            )

        if (step + 1) % grad_accum_steps == 0:
            actual_commits = commits_by_step.pop(step, {})
            if set(actual_commits) != set(accumulated):
                raise AssertionError(
                    f"Sparse commit coverage mismatch step={step} "
                    f"missing={sorted(set(accumulated) - set(actual_commits))[:8]} "
                    f"extra={sorted(set(actual_commits) - set(accumulated))[:8]}"
                )
            for physical_key, grad in accumulated.items():
                committed_gradient, committed_after = actual_commits[physical_key]
                np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
                    committed_gradient,
                    grad,
                    rtol=SPARSE_NUMERICAL_RTOL,
                    atol=SPARSE_NUMERICAL_ATOL,
                    err_msg=(
                        f"Sparse accumulated gradient mismatch step={step} "
                        f"physical_key={physical_key}"
                    ),
                )
                current_before = current_embeddings[physical_key]
                m, v, adam_step = adam_state.get(
                    physical_key,
                    (np.zeros_like(grad), np.zeros_like(grad), 0),
                )
                adam_step += 1
                m = np.float32(0.9) * m + np.float32(0.1) * grad  # pyrefly: ignore[unsupported-operation]
                v = np.float32(0.999) * v + np.float32(0.001) * np.square(grad)  # pyrefly: ignore[no-matching-overload]
                m_hat = m / np.float32(1.0 - 0.9**adam_step)
                v_hat = v / np.float32(1.0 - 0.999**adam_step)
                current_embeddings[physical_key] = current_before - np.float32(
                    1e-3
                ) * m_hat / (np.sqrt(v_hat) + np.float32(1e-8))
                committed_versions[physical_key].append(
                    (step, current_embeddings[physical_key].copy())
                )
                adam_state[physical_key] = (m, v, adam_step)
                np.testing.assert_allclose(
                    committed_after,
                    current_embeddings[physical_key],
                    rtol=SPARSE_NUMERICAL_RTOL,
                    atol=SPARSE_NUMERICAL_ATOL,
                    err_msg=(
                        f"Sparse committed state mismatch step={step} "
                        f"physical_key={physical_key}"
                    ),
                )
                update_checks += 1
            accumulated.clear()
        elif step in commits_by_step:
            raise AssertionError(
                f"Sparse commit occurred off optimizer boundary: {step}"
            )

        for physical_key, _, _, value_after in entries_by_step[step]:
            np.testing.assert_allclose(
                value_after,
                current_embeddings[physical_key],
                rtol=SPARSE_NUMERICAL_RTOL,
                atol=SPARSE_NUMERICAL_ATOL,
                err_msg=(
                    f"Sparse committed state mismatch step={step} "
                    f"physical_key={physical_key}"
                ),
            )
    if set(dense_rows) != set(update_rows):
        raise AssertionError("Dense gradient -> Sparse update coverage mismatch")
    if accumulated:
        raise AssertionError("Sparse gradients remain after final optimizer boundary")
    if commits_by_step:
        raise AssertionError(
            f"Sparse commit trace has unexpected steps: {sorted(commits_by_step)}"
        )
    if update_checks == 0:
        raise AssertionError("no Sparse embedding parameter update was verified")
    return len(update_rows), update_checks, stale_lookup_checks


def audit_dense_updates(records, dense_world_size, expected_steps, grad_accum_steps):
    expected = {
        (step, rank)
        for step in range(expected_steps)
        for rank in range(dense_world_size)
    }
    actual = {(int(row["step"]), int(row["rank"])) for row in records}
    if actual != expected:
        raise AssertionError(f"Dense update coverage mismatch: {actual ^ expected}")
    by_step = defaultdict(list)
    for record in records:
        step = int(record["step"])
        payload = record["payload"]
        expected_micro = step % grad_accum_steps + 1
        expected_applied = expected_micro == grad_accum_steps
        if payload["micro_step"] != expected_micro:
            raise AssertionError(f"Dense micro-step mismatch: {record}")
        if bool(payload["optimizer_applied"]) != expected_applied:
            raise AssertionError(f"Dense optimizer boundary mismatch: {record}")
        before = payload["model_before"]["fingerprint"]
        after = payload["model_after"]["fingerprint"]
        if (before != after) != expected_applied:
            raise AssertionError(f"Dense parameter transition mismatch: {record}")
        by_step[step].append((before, after))
    for step, values in by_step.items():
        if len({before for before, _after in values}) != 1:
            raise AssertionError(f"Dense ranks disagree before step {step}")
        if len({after for _before, after in values}) != 1:
            raise AssertionError(f"Dense ranks disagree after step {step}")
    for step in range(expected_steps - 1):
        if by_step[step][0][1] != by_step[step + 1][0][0]:
            raise AssertionError(f"Dense state discontinuity after step {step}")
    return sum(bool(row["payload"]["optimizer_applied"]) for row in records)


def audit_tensorboard(logdir, target_names, expected_steps):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars")
    if not isinstance(scalar_tags, list):
        raise TypeError("TensorBoard event log has no scalar tag list")
    tags = set(scalar_tags)
    required = {
        "train/loss",
        "train/throughput_samples_per_sec",
        "train/throughput_steps_per_sec",
    }
    for target in target_names:
        required.update(
            {
                f"train/auc_defined_{target}",
                f"train/gauc_defined_{target}",
                f"train/calibration_defined_{target}",
            }
        )
    if required - tags:
        raise AssertionError(f"missing TensorBoard tags: {sorted(required - tags)}")
    expected_event_steps = list(range(1, expected_steps + 1))
    values = {}
    for tag in sorted(required):
        events = accumulator.Scalars(tag)
        if [int(event.step) for event in events] != expected_event_steps:
            raise AssertionError(f"TensorBoard step mismatch tag={tag}: {events}")
        tag_values = [float(event.value) for event in events]
        if not np.all(np.isfinite(tag_values)):
            raise AssertionError(f"non-finite TensorBoard values tag={tag}")
        if "_defined_" in tag and any(value not in (0.0, 1.0) for value in tag_values):
            raise AssertionError(f"invalid AUC/GAUC defined flag tag={tag}")
        values[tag] = tag_values
    for target in target_names:
        for metric in ("auc", "gauc", "calibration"):
            defined_tag = f"train/{metric}_defined_{target}"
            value_tag = f"train/{metric}_{target}"
            defined_steps = [
                step
                for step, defined in zip(
                    expected_event_steps, values[defined_tag], strict=True
                )
                if defined == 1.0
            ]
            events = accumulator.Scalars(value_tag) if value_tag in tags else []
            actual_steps = [int(event.step) for event in events]
            if actual_steps != defined_steps:
                raise AssertionError(
                    f"TensorBoard {metric.upper()} defined/value mismatch "
                    f"target={target} defined_steps={defined_steps} "
                    f"value_steps={actual_steps}"
                )
            metric_values = [float(event.value) for event in events]
            invalid_range = metric in ("auc", "gauc") and any(
                value < 0.0 or value > 1.0 for value in metric_values
            )
            if not np.all(np.isfinite(metric_values)) or invalid_range:
                raise AssertionError(
                    f"invalid TensorBoard {metric.upper()} values "
                    f"target={target}: {metric_values}"
                )
            values[value_tag] = metric_values
    if len(set(values["train/loss"])) < 2:
        raise AssertionError("loss did not change across the optimizer boundary")
    return values


def main() -> None:
    args = parse_args()
    feature_layouts, target_names = load_model_contract(args.conf_version)
    workers = parse_sparse_workers(args.sparse_workers, tuple(feature_layouts))
    records = load_records(args.trace_dir)
    input_rows, feature_counts, occurrence_count = audit_input(
        records["input_unique"],
        feature_layouts,
        args.dense_world_size,
        args.expected_steps,
    )
    lookup_rows, rank_coverage = audit_lookup(
        records["sparse_lookup"], workers, input_rows
    )
    gather_rows = audit_dense_forward(
        records["dense_gather"], lookup_rows, args.dense_world_size
    )
    gradient_rows, sparse_parameter_updates, stale_sparse_lookups = (
        audit_gradient_and_sparse_update(
            records["dense_grad"],
            records["sparse_update"],
            records["sparse_commit"],
            lookup_rows,
            args.grad_accum_steps,
            args.sparse_max_staleness_steps,
        )
    )
    dense_parameter_updates = audit_dense_updates(
        records["dense_update"],
        args.dense_world_size,
        args.expected_steps,
        args.grad_accum_steps,
    )
    tensorboard = audit_tensorboard(
        args.tensorboard_dir, target_names, args.expected_steps
    )
    result = {
        "passed": True,
        "conf_version": args.conf_version,
        "steps": args.expected_steps,
        "dense_world_size": args.dense_world_size,
        "sparse_workers": workers,
        "fid_feature_count": len(feature_counts),
        "fid_feature_occurrences": feature_counts,
        "fid_occurrences_checked": occurrence_count,
        "compact_rows_checked": len(input_rows),
        "embedding_rows_checked": len(gather_rows),
        "gradient_rows_checked": gradient_rows,
        "sparse_parameter_updates_checked": sparse_parameter_updates,
        "bounded_stale_sparse_lookups_checked": stale_sparse_lookups,
        "dense_parameter_updates_checked": dense_parameter_updates,
        "sparse_rank_coverage": {
            f"{group}:step{step}": sorted(ranks)
            for (group, step), ranks in sorted(rank_coverage.items())
        },
        "tensorboard": tensorboard,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
