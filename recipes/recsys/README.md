# Recommendation Recipes

Reusable recommendation components built on Flare, Fluxway, JAX and native
sparse storage. Application adapters supply model configuration, label maps,
feature manifests and sources. The shared package has no application imports
or configuration-file discovery.

## Use From a Source Checkout

Install Flare and the additional recipe dependencies in the same environment:

```sh
python -m pip install -e '.[dev]'
python -m pip install -r recipes/recsys/requirements.txt
JAX_PLATFORMS=cpu python -m recipes.recsys.examples.synthetic_ranking
python -m pytest -q
```

`recipes.recsys` is distributed in the source tree, separately from the
`flare` wheel. Run these commands from the checkout root. The synthetic
example trains and evaluates a small dense model locally; it does not start a
distributed runtime or access external data.

## Composition

| Layer | Responsibility |
| --- | --- |
| `flare` | Data, execution, mesh, transport and checkpoint contracts |
| `recipes.recsys` | Recommendation objectives, tasks, graph assembly, workers, metrics and benchmarks |
| Application adapter | Model factory, ordered targets, private configuration decoding, feature/source bindings and placement exclusions |

An adapter constructs a `RankingTask` with an objective, a label column and a
grouping column. Model target order is explicit and shared by training loss
and evaluation metrics. New applications provide their own adapter and model
factory without changing the shared graph.

See [ranking](ranking/README.md) for graph composition, placement, sparse
schemas, valid-row masking and immutable inference weights.
