import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from recipes.recsys.ranking.losses import MultiTargetCrossEntropyLoss


def test_create_maps_sample_weights_to_labels_weights_and_class_weights():
    loss = MultiTargetCrossEntropyLoss.create(
        sample_weight_map={
            "target_a": {
                "class_weight": 0.5,
                "sample_weight": {
                    "dt_0": 1.0,
                    "dt_1": -2.0,
                },
            },
            "target_b": {
                "class_weight": 2.0,
                "sample_weight": {
                    "dt_1": 1.5,
                    "dt_2": -1.0,
                },
            },
        },
        label_map={"dt_0": 0, "dt_1": 1, "dt_2": 2},
        target_names=("target_a", "target_b"),
    )

    assert loss.targets == ("target_a", "target_b")
    np.testing.assert_array_equal(
        np.asarray(loss.target_labels),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(loss.target_weights),
        np.array([[1.0, 2.0, 0.0], [0.0, 1.5, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(loss.class_weights),
        np.array([0.5, 2.0], dtype=np.float32),
    )


def test_call_matches_weighted_sigmoid_binary_cross_entropy():
    loss = MultiTargetCrossEntropyLoss.create(
        sample_weight_map={
            "target_a": {
                "class_weight": 0.5,
                "sample_weight": {
                    "dt_0": 1.0,
                    "dt_1": -2.0,
                },
            },
            "target_b": {
                "class_weight": 2.0,
                "sample_weight": {
                    "dt_1": 1.5,
                    "dt_2": -1.0,
                },
            },
        },
        label_map={"dt_0": 0, "dt_1": 1, "dt_2": 2},
        target_names=("target_a", "target_b"),
    )
    logits = jnp.array(
        [
            [0.5, -0.25],
            [1.0, 0.0],
            [-1.0, 0.75],
            [0.25, -0.5],
        ],
        dtype=jnp.float32,
    )
    label_ids = jnp.array([0, 1, 2, 1], dtype=jnp.int32)

    row_weights = jnp.full((4,), 0.25, dtype=jnp.float32)
    actual = loss(logits, label_ids, row_weights)
    labels = loss.target_labels[:, label_ids].T
    weights = loss.target_weights[:, label_ids].T * loss.class_weights
    expected = jnp.mean(
        jnp.sum(optax.sigmoid_binary_cross_entropy(logits, labels) * weights, axis=-1)
    )

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6
    )


def test_padding_rows_have_zero_gradient_with_normalized_row_weights() -> None:
    loss = MultiTargetCrossEntropyLoss.create(
        sample_weight_map={
            "target": {
                "sample_weight": {
                    "negative": -1.0,
                    "positive": 1.0,
                }
            }
        },
        label_map={"negative": 0, "positive": 1},
        target_names=("target",),
    )
    real_logits = jnp.asarray([[0.5], [-0.25]], dtype=jnp.float32)
    padded_logits = jnp.asarray([[0.5], [-0.25], [-0.25], [-0.25]], dtype=jnp.float32)
    real_labels = jnp.asarray([1, 0], dtype=jnp.int32)
    padded_labels = jnp.asarray([1, 0, 0, 0], dtype=jnp.int32)

    real_gradient = jax.grad(
        lambda logits: loss(
            logits,
            real_labels,
            jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        )
    )(real_logits)
    padded_gradient = jax.grad(
        lambda logits: loss(
            logits,
            padded_labels,
            jnp.asarray([0.5, 0.5, 0.0, 0.0], dtype=jnp.float32),
        )
    )(padded_logits)

    np.testing.assert_allclose(padded_gradient[:2], real_gradient)
    np.testing.assert_array_equal(padded_gradient[2:], np.zeros((2, 1)))


def test_valid_mask_zeroes_padding_gradient() -> None:
    loss = MultiTargetCrossEntropyLoss.create(
        sample_weight_map={
            "target": {
                "sample_weight": {
                    "negative": -1.0,
                    "positive": 1.0,
                }
            }
        },
        label_map={"negative": 0, "positive": 1},
        target_names=("target",),
    )
    real_logits = jnp.asarray([[0.5], [-0.25]], dtype=jnp.float32)
    padded_logits = jnp.asarray([[0.5], [-0.25], [-0.25], [-0.25]], dtype=jnp.float32)
    real_labels = jnp.asarray([1, 0], dtype=jnp.int32)
    padded_labels = jnp.asarray([1, 0, 0, 0], dtype=jnp.int32)

    real_gradient = jax.grad(
        lambda logits: loss(
            logits,
            real_labels,
            jnp.asarray([1.0, 1.0], dtype=jnp.float32),
            valid_mask=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        )
    )(real_logits)
    padded_gradient = jax.grad(
        lambda logits: loss(
            logits,
            padded_labels,
            jnp.asarray([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32),
            valid_mask=jnp.asarray([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
        )
    )(padded_logits)

    np.testing.assert_allclose(padded_gradient[:2], real_gradient)
    np.testing.assert_array_equal(padded_gradient[2:], np.zeros((2, 1)))


@pytest.mark.parametrize("weight", [1.0, 0.0])
def test_create_rejects_unknown_label_reference(weight):
    with pytest.raises(KeyError, match="missing"):
        MultiTargetCrossEntropyLoss.create(
            sample_weight_map={
                "target": {
                    "sample_weight": {
                        "missing": weight,
                    },
                },
            },
            label_map={"dt_0": 0},
            target_names=("target",),
        )


@pytest.mark.parametrize("use_valid_mask", [False, True])
def test_zero_sample_weight_has_zero_loss_and_gradient(use_valid_mask):
    loss = MultiTargetCrossEntropyLoss.create(
        sample_weight_map={
            "target": {"sample_weight": {"ignored": 0.0, "positive": 1.0}},
        },
        label_map={"ignored": 0, "positive": 1},
        target_names=("target",),
    )
    logits = jnp.asarray([[3.0], [-2.0]], dtype=jnp.float32)
    labels = jnp.asarray([0, 0], dtype=jnp.int32)
    row_weights = jnp.asarray([0.5, 0.5], dtype=jnp.float32)
    valid_mask = jnp.ones((2,), dtype=jnp.float32) if use_valid_mask else None
    value, gradient = jax.value_and_grad(
        lambda values: loss(values, labels, row_weights, valid_mask=valid_mask)
    )(logits)
    np.testing.assert_array_equal(value, 0.0)
    np.testing.assert_array_equal(gradient, np.zeros((2, 1), dtype=np.float32))


def test_zero_weight_masks_only_the_configured_target_and_label():
    loss = MultiTargetCrossEntropyLoss.create(
        sample_weight_map={
            "reply": {
                "sample_weight": {"ignored": 3.0, "negative": 0.0, "positive": -4.0}
            },
            "engagement": {
                "sample_weight": {"ignored": 0.0, "negative": -2.0, "positive": 1.0}
            },
        },
        label_map={"ignored": 0, "negative": 1, "positive": 2},
        target_names=("engagement", "reply"),
    )
    value, gradient = jax.value_and_grad(
        lambda values: loss(values, jnp.asarray([0, 1, 2]), jnp.ones((3,)))
    )(jnp.zeros((3, 2), dtype=jnp.float32))
    np.testing.assert_allclose(value, 10 * np.log(2), rtol=1e-6)
    np.testing.assert_array_equal(gradient, [[0, -1.5], [1, 0], [-0.5, 2]])
    np.testing.assert_array_equal(loss.target_labels, [[0, 0, 1], [1, 0, 0]])


@pytest.mark.parametrize("weight", [np.nan, np.inf, -np.inf])
def test_create_rejects_non_finite_sample_weight(weight):
    with pytest.raises(ValueError, match="finite"):
        MultiTargetCrossEntropyLoss.create(
            sample_weight_map={
                "target": {
                    "sample_weight": {
                        "dt_0": weight,
                    },
                },
            },
            label_map={"dt_0": 0},
            target_names=("target",),
        )


@pytest.mark.parametrize("weight", [np.nan, np.inf, -np.inf])
def test_create_rejects_non_finite_class_weight(weight):
    with pytest.raises(ValueError, match="Class weight.*finite"):
        MultiTargetCrossEntropyLoss.create(
            sample_weight_map={
                "target": {
                    "class_weight": weight,
                    "sample_weight": {"dt_0": 1.0},
                },
            },
            label_map={"dt_0": 0},
            target_names=("target",),
        )


def test_target_order_is_explicit_and_shared_with_label_weights():
    loss = MultiTargetCrossEntropyLoss.create(
        label_map={"negative": 0, "positive": 1},
        target_names=("reply", "engagement"),
        sample_weight_map={
            "engagement": {"sample_weight": {"negative": -2.0, "positive": 1.0}},
            "reply": {"sample_weight": {"negative": 3.0, "positive": -4.0}},
        },
    )
    assert loss.targets == ("reply", "engagement")
    np.testing.assert_array_equal(loss.target_labels, [[1, 0], [0, 1]])
    np.testing.assert_array_equal(loss.target_weights, [[3, 4], [2, 1]])
    logits = jnp.asarray([[3.0, -3.0]], dtype=jnp.float32)
    expected = 3 * optax.sigmoid_binary_cross_entropy(logits[0, 0], 1)
    expected += 2 * optax.sigmoid_binary_cross_entropy(logits[0, 1], 0)
    np.testing.assert_allclose(loss(logits, jnp.asarray([0]), jnp.ones((1,))), expected)
