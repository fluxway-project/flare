from __future__ import annotations

from recipes.recsys.ranking.metrics.config import build_auc_metric_specs


def test_build_auc_metric_specs_is_domain_named() -> None:
    specs = build_auc_metric_specs(("click", "reply"))

    assert tuple(spec.name for spec in specs) == (
        "auc_click",
        "gauc_click",
        "auc_reply",
        "gauc_reply",
    )
    assert all(spec.higher_is_better for spec in specs)
