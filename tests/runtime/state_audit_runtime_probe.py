from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from flare.runtime.diagnostics.state_audit import fingerprint_state

mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("dp",))
value = jax.make_array_from_single_device_arrays(
    (4,),
    NamedSharding(mesh, P("dp")),
    [
        jax.device_put(jnp.asarray([1.0, jnp.nan]), jax.devices()[0]),
        jax.device_put(jnp.asarray([jnp.inf, -2.0]), jax.devices()[1]),
    ],
)
fingerprint = fingerprint_state({"value": value})
assert fingerprint["element_count"] == 4
assert fingerprint["finite_count"] == 2
assert fingerprint["nan_count"] == 1
assert fingerprint["inf_count"] == 1
assert fingerprint["non_finite_leaves"] == [
    {
        "leaf_index": 0,
        "path": "['value']",
        "shape": [4],
        "finite_count": 2,
        "nan_count": 1,
        "inf_count": 1,
    }
]
assert np.isnan(fingerprint["sum"])
assert np.isnan(fingerprint["abs_sum"])
print("STATE_AUDIT_RUNTIME=PASS")
