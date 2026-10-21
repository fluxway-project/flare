"""Data schemas and feature-layout contracts.

Schema modules are imported explicitly to keep optional runtime dependencies
out of the lightweight feature-catalog and layout APIs.
"""

from flare.data.schema.batch import RaggedFeatureBatch
from flare.data.schema.feature_partition import (
    FeaturePartition,
    compile_feature_partition,
)
from flare.data.schema.transform import (
    NULL_FID_SENTINEL,
    table_to_training_dict,
)
from flare.data.schema.virtual_pad import (
    DEFAULT_VALIDITY_COLUMN,
    ArrowVirtualPadBatchFactory,
)

__all__ = [
    "DEFAULT_VALIDITY_COLUMN",
    "NULL_FID_SENTINEL",
    "ArrowVirtualPadBatchFactory",
    "FeaturePartition",
    "RaggedFeatureBatch",
    "compile_feature_partition",
    "table_to_training_dict",
]
