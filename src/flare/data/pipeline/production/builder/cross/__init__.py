"""Cross-feature subpackage — protocols, registry, and concrete Expr operators."""

from flare.data.pipeline.production.builder.cross.base import (
    get_fea_factory,
    get_fid_factory,
    register_fea,
    register_fid,
)
from flare.data.pipeline.production.builder.cross.fea_ops import (
    DivCrossExpr,
    HitCountCrossExpr,
    SubCrossExpr,
)
from flare.data.pipeline.production.builder.cross.fid_ops import (
    FidCrossExpr as FidCrossNExpr,
)
from flare.data.pipeline.production.builder.cross.protocol import (
    FeaCrossExpr,
    FidCrossExpr,
)

__all__ = [
    "DivCrossExpr",
    "FeaCrossExpr",
    "FidCrossExpr",
    "FidCrossNExpr",
    "HitCountCrossExpr",
    "SubCrossExpr",
    "get_fea_factory",
    "get_fid_factory",
    "register_fea",
    "register_fid",
]
