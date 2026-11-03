from __future__ import annotations

import os
from decimal import Decimal
from itertools import pairwise

import numpy as np
import pytest

from flare.runtime.fluxway.resources import normalize_resource_vector
from recipes.recsys.ranking.metrics.ray_data import (
    _METRIC_PARTIAL_DTYPE,
    RayDataMetricsComputer,
    _merge_auc_partition_rows,
    _merge_grouped_auc_partitions,
    _serialize_sorted_metrics_partition,
    _summarize_sorted_metrics_partition,
)


@pytest.fixture(scope="module", params=(None, 0.25))
def metric_computation(request):
    import ray
    import ray.data

    address = os.environ.get("FLARE_METRICS_TEST_RAY_ADDRESS")
    if not address:
        pytest.skip("requires an explicit existing FLARE_METRICS_TEST_RAY_ADDRESS")
    if ray.is_initialized():
        raise RuntimeError("Metrics integration test requires its own Ray client")
    ray.init(address=address, namespace="flare-metrics-regression", log_to_driver=False)
    try:
        target_names = tuple(f"target_{index}" for index in range(6))
        base_labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        base_scores = np.asarray(
            [0.1, 0.5, 0.5, 0.9, 0.2, 0.5, 0.5, 0.8, 0.3, 0.5, 0.5, 0.7],
            dtype=np.float32,
        )
        labels = np.stack([np.roll(base_labels, index) for index in range(6)], axis=1)
        scores = np.stack([np.roll(base_scores, index) for index in range(6)], axis=1)
        groups = np.asarray([10, 20, 30] * 4, dtype=np.int64)
        limits = {"CPU": Decimal(2), "memory": Decimal(2 * 1024**3)}
        computer = RayDataMetricsComputer(
            target_names=target_names,
            negative_factor=request.param,
            ray_component_limits=limits,
            resource_normalizer=normalize_resource_vector,
        )
        cuts = (0, 2, 5, 6, 10, 12)
        refs = tuple(
            computer.put_batch(
                scores=scores[start:end],
                labels=labels[start:end],
                groups=groups[start:end],
            )
            for start, end in pairwise(cuts)
        )
        datasets = []
        projection_blocks = []
        from_arrow_refs = ray.data.from_arrow_refs
        select_columns = ray.data.Dataset.select_columns

        def record_dataset(values):
            dataset = from_arrow_refs(values)
            datasets.append(dataset)
            return dataset

        def record_projection(dataset, *args, **kwargs):
            projection_blocks.append(dataset.num_blocks())
            return select_columns(dataset, *args, **kwargs)

        context = ray.data.DataContext.get_current()
        previous = context.execution_options.resource_limits.copy()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ray.data, "from_arrow_refs", record_dataset)
            patch.setattr(ray.data.Dataset, "select_columns", record_projection)
            results = computer.compute(refs)
        assert len(datasets) == 1
        yield {
            "limits": datasets[0].context.execution_options.resource_limits,
            "global_limits_before": previous,
            "global_limits_after": context.execution_options.resource_limits.copy(),
            "projection_blocks": projection_blocks,
            "results": results,
            "target_names": target_names,
            "labels": labels,
            "scores": scores,
            "groups": groups,
            "negative_factor": request.param,
        }
    finally:
        ray.shutdown()


def test_dataset_captures_component_limits_and_restores_global_context(
    metric_computation,
) -> None:
    computation = metric_computation
    assert computation["limits"].cpu == 2
    assert computation["limits"].memory == 2 * 1024**3
    assert computation["global_limits_after"] == computation["global_limits_before"]


def test_target_projections_reuse_coalesced_blocks(
    metric_computation,
) -> None:
    assert metric_computation["projection_blocks"] == [2] * 6


def test_metrics_match_full_dataset_reference_with_ties_and_split_groups(
    metric_computation,
) -> None:
    from sklearn.metrics import roc_auc_score

    computation = metric_computation
    groups = computation["groups"]
    for index, name in enumerate(computation["target_names"]):
        labels = computation["labels"][:, index]
        scores = computation["scores"][:, index]
        result = computation["results"][name]
        assert result["sample_count"] == len(labels)
        assert result["auc_defined"] is True
        assert result["auc"] == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
        weighted_auc = 0.0
        total_weight = 0.0
        valid_groups = 0
        for group in np.unique(groups):
            selected = groups == group
            group_labels = labels[selected]
            positive = int(group_labels.sum())
            negative = len(group_labels) - positive
            if not positive or not negative:
                continue
            weight = float(positive)
            if computation["negative_factor"] is not None:
                weight += negative * computation["negative_factor"]
            weighted_auc += roc_auc_score(group_labels, scores[selected]) * weight
            total_weight += weight
            valid_groups += 1
        assert result["grouped_auc_defined"] is True
        assert result["grouped_auc"] == pytest.approx(
            weighted_auc / total_weight, abs=1e-12
        )
        assert result["grouped_auc_valid_group_count"] == valid_groups
        assert result["grouped_auc_weight_sum"] == total_weight


@pytest.mark.parametrize("negative_factor", [None, 0.0, 0.25, 2.0])
def test_segmented_group_auc_matches_sklearn_across_score_partitions(
    negative_factor,
) -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(7)
    # One large group, many small groups, ties spanning all partitions, and
    # single-class groups exercise the exact cross-partition merge.
    groups = np.concatenate(
        (np.full(400, -10, dtype=np.int64), rng.integers(0, 150, 600))
    )
    labels = rng.integers(0, 2, len(groups), dtype=np.int8)
    labels[groups == 0] = 0
    labels[groups == 1] = 1
    scores = rng.integers(0, 11, len(groups)).astype(np.float32) / 10
    order = np.argsort(scores, kind="stable")
    groups, labels, scores = groups[order], labels[order], scores[order]
    partial_rows = []
    for start, end in zip((0, 130, 400, 750), (130, 400, 750, 1000)):
        batch = {
            "group_key": groups[start:end],
            "label": labels[start:end],
            "score": scores[start:end],
        }
        summary = _summarize_sorted_metrics_partition(
            batch,
            label_column="label",
            score_column="score",
        )
        serialized = _serialize_sorted_metrics_partition(
            batch, label_column="label", score_column="score"
        )
        packed = np.frombuffer(
            serialized["summaries"][0].as_py(), dtype=_METRIC_PARTIAL_DTYPE
        )
        for name in _METRIC_PARTIAL_DTYPE.names or ():
            np.testing.assert_array_equal(packed[name], summary[name])
        partial_rows.extend(packed)
    global_result = _merge_auc_partition_rows(
        [row for row in partial_rows if row["is_global"]]
    )
    grouped_result = _merge_grouped_auc_partitions(
        (row for row in partial_rows if not row["is_global"]),
        negative_factor=negative_factor,
        target_name="target",
    )
    assert global_result["auc"] == pytest.approx(
        roc_auc_score(labels, scores), abs=1e-12
    )
    expected_weighted_auc = 0.0
    expected_weight = 0.0
    expected_valid = 0
    for group in np.unique(groups):
        selected = groups == group
        group_labels = labels[selected]
        positive = int(group_labels.sum())
        negative = len(group_labels) - positive
        if not positive or not negative:
            continue
        weight = float(positive)
        if negative_factor is not None:
            weight += negative * negative_factor
        expected_weighted_auc += roc_auc_score(group_labels, scores[selected]) * weight
        expected_weight += weight
        expected_valid += 1
    assert grouped_result["grouped_auc"] == pytest.approx(
        expected_weighted_auc / expected_weight, abs=1e-12
    )
    assert grouped_result["grouped_auc_valid_group_count"] == expected_valid
    assert grouped_result["grouped_auc_weight_sum"] == expected_weight
