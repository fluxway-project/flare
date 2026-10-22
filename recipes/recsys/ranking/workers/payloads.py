from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, make_dataclass
from typing import Any
from weakref import WeakSet

import jax.tree_util as jtu
from fluxway.common.decorators import data_type
from fluxway.common.types.tensor import TransportTensor

from flare.training.contracts.payloads import PayloadSchema
from recipes.recsys.ranking.schema.sparse_features import parse_sparse_group_name


class _JaxDataTypeMeta(type):
    _registered: WeakSet[type] = WeakSet()

    def __setattr__(cls, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name != "__dataclass_fields__" or cls in cls._registered:
            return
        jtu.register_dataclass(
            cls,
            data_fields=tuple(value),
            meta_fields=(),
        )
        cls._registered.add(cls)


class _JaxDataType(metaclass=_JaxDataTypeMeta):
    __slots__ = ()


def build_fluxway_dataclass(
    name: str,
    fields: Sequence[tuple[str, type]],
) -> type:
    cls = make_dataclass(
        name,
        list(fields),
        bases=(_JaxDataType,),
        namespace={"__module__": __name__},
    )
    return data_type()(cls)


def compile_payload_schema_to_fluxway(
    schema: PayloadSchema,
) -> type:
    fields = [(field.name, TransportTensor) for field in schema.fields]
    return build_fluxway_dataclass(schema.name, fields)


@dataclass(frozen=True)
class DynamicDataTypes:
    ranking_reader_output: type
    input_node_output: type
    lookup_outputs: Mapping[str, type]
    ranking_train_output: type
    ranking_eval_output: type
    sparse_groups: tuple[str, ...]

    def require(self, name: str) -> type:
        fixed = {
            "RankingReaderOutput": self.ranking_reader_output,
            "InputNodeOutput": self.input_node_output,
            "RankingTrainOutput": self.ranking_train_output,
            "RankingEvalOutput": self.ranking_eval_output,
        }
        if name in fixed:
            return fixed[name]
        prefix = "LookupOutput_"
        if name.startswith(prefix):
            value = self.lookup_outputs.get(name[len(prefix) :])
            if value is not None:
                return value
        raise KeyError(f"unknown dynamic data type {name!r}")


def build_ranking_fluxway_data_types(
    sparse_groups: tuple[str, ...],
    batch_fields: tuple[str, ...] = (
        "side_info",
        "labels",
        "group_id",
        "samples_valid_count",
    ),
) -> DynamicDataTypes:
    fields = [(name, TransportTensor) for name in sparse_groups] + [
        (name, TransportTensor) for name in batch_fields
    ]
    ranking_reader_output = build_fluxway_dataclass("RankingReaderOutput", fields)

    input_fields: list[tuple[str, type]] = []
    for name in sparse_groups:
        input_fields.extend(
            (
                (f"unique_{name}", TransportTensor),
                (f"inv_{name}", TransportTensor),
            )
        )
    input_fields.extend((name, TransportTensor) for name in batch_fields)
    input_node_output = build_fluxway_dataclass("InputNodeOutput", input_fields)

    groups_by_base: dict[str, list[str]] = {}
    for name in sparse_groups:
        base_name, _ = parse_sparse_group_name(name)
        groups_by_base.setdefault(base_name, []).append(name)
    lookup_outputs = {
        base_name: build_fluxway_dataclass(
            f"LookupOutput_{base_name}",
            [(f"sparse_features_{name}", TransportTensor) for name in names],
        )
        for base_name, names in groups_by_base.items()
    }

    model_out_fields = [(f"grad_{name}", TransportTensor) for name in sparse_groups]
    model_out_fields.extend(
        (
            ("scores", TransportTensor),
            ("loss", TransportTensor),
            ("metric_sampled", TransportTensor),
        )
    )
    return DynamicDataTypes(
        ranking_reader_output=ranking_reader_output,
        input_node_output=input_node_output,
        lookup_outputs=lookup_outputs,
        ranking_train_output=build_fluxway_dataclass(
            "RankingTrainOutput", model_out_fields
        ),
        ranking_eval_output=build_fluxway_dataclass(
            "RankingEvalOutput",
            [("scores", TransportTensor), ("labels", TransportTensor)],
        ),
        sparse_groups=sparse_groups,
    )


__all__ = [
    "DynamicDataTypes",
    "build_fluxway_dataclass",
    "build_ranking_fluxway_data_types",
    "compile_payload_schema_to_fluxway",
]
