from dataclasses import dataclass
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pytest

from flare.data.schema.feature_catalog import FeatureOutputKind
from flare.data.schema.feature_layout import TrainingFeatureLayout
from recipes.recsys.ranking.losses import MultiTargetCrossEntropyLoss
from recipes.recsys.ranking.recipe import BaseRankingRecipe
from recipes.recsys.ranking.task import RankingTask


@pytest.fixture
def objective():
    return MultiTargetCrossEntropyLoss.create(
        label_map={"ignored": 0, "selected": 1},
        target_names=("engagement",),
        sample_weight_map={
            "engagement": {"sample_weight": {"ignored": -2.0, "selected": 1.0}},
        },
    )


@pytest.mark.parametrize(
    "label_column,group_column", [("outcome", "visitor"), ("feedback", "session")]
)
def test_task_maps_application_columns_to_shared_labels(
    objective, label_column, group_column
):
    task = RankingTask(label_column, group_column, objective)
    layout = TrainingFeatureLayout(
        projection=("outcome_values", "group_values"),
        bindings=((label_column, "outcome_values"), (group_column, "group_values")),
        kinds_by_name={
            label_column: FeatureOutputKind.DENSE_I32_INDICES,
            group_column: FeatureOutputKind.SPARSE_I64_FIDS,
        },
    )
    transform = task.batch_transformer_factory(
        side_info_seq=(), categorical_feature_groups={}
    )(layout)
    batch = transform(
        pa.table(
            {
                "outcome_values": pa.array([1, 0], type=pa.int32()),
                "group_values": pa.array([9, 9], type=pa.int64()),
            }
        )
    )
    np.testing.assert_array_equal(batch["labels"], [1, 0])
    np.testing.assert_array_equal(batch[group_column], [9, 9])
    assert label_column not in batch


class SquaredErrorObjective:
    targets = ("engagement",)
    target_labels = np.asarray([[0, 1]], dtype=np.float32)

    def __call__(self, logits, labels, row_weights):
        return jnp.sum((logits[:, 0] - labels) ** 2 * row_weights)


@dataclass(frozen=True)
class SyntheticRecipe(BaseRankingRecipe):
    task: RankingTask | None = None

    def ranking_task(self):
        if self.task is None:
            raise ValueError("a task is required")
        return self.task

    def model_builder(self):
        return SimpleNamespace(
            pack_meta=SimpleNamespace(sparse_id_groups={}, side_info_seq=()),
            create_dummy_inputs=lambda **kwargs: ({"side_info": jnp.zeros((2, 0))}, {}),
        )


@pytest.mark.parametrize("objective_factory", [SquaredErrorObjective, None])
def test_recipe_accepts_an_injected_objective_without_application_config(
    objective, objective_factory
):
    selected = objective if objective_factory is None else objective_factory()
    recipe = SyntheticRecipe(
        name="synthetic",
        task_name="engagement",
        batch_size=2,
        task=RankingTask("outcome", "visitor", selected),
    )
    spec = recipe.task_spec()
    assert set(spec.batch_schema.fields) == {"side_info", "labels", "row_weights"}
    assert spec.output_ports[0].spec.shape == (2, 1)
    batches = list(recipe.training_batches(batch_size=2, steps=2))
    assert len(batches) == 2
    _, targets = batches[0]
    np.testing.assert_array_equal(targets["labels"], [0, 1])
    logits = jnp.asarray([[0.25], [0.75]], dtype=jnp.float32)
    actual = spec.loss({"logits": logits}, targets)
    expected = selected(logits, targets["labels"], targets["row_weights"]) / 2
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "label,group", [("", "visitor"), ("outcome", ""), ("same", "same")]
)
def test_task_rejects_invalid_column_bindings(objective, label, group):
    with pytest.raises(ValueError):
        RankingTask(label, group, objective)
