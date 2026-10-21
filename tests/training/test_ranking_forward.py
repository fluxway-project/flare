from __future__ import annotations

import pytest

from recipes.recsys.ranking.graph.forward import ranking_forward


def test_ranking_forward_requires_the_recipe_output_contract() -> None:
    assert ranking_forward(
        lambda inputs: {"logits": inputs["feature"]}, {"feature": 3}
    ) == {"logits": 3}
    with pytest.raises(KeyError, match="logits"):
        ranking_forward(lambda inputs: {}, {})
    with pytest.raises(TypeError, match="Mapping"):
        ranking_forward(lambda inputs: 1, {})
