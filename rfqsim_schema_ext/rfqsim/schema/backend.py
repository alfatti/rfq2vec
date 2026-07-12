"""CuPy/NumPy backend switch.

Import `xp` from here everywhere in the production path. On the H200 box
CuPy is present and all array work runs on device; in a CPU environment the
same code runs on NumPy, byte-identically for integer/RNG work and to float
rounding for the algebra. One process per GPU: set CUDA_VISIBLE_DEVICES per
shard worker; nothing in this package shares device state across processes.
"""
from __future__ import annotations

import numpy as _np

try:  # pragma: no cover - exercised only on GPU hosts
    import cupy as xp
    _GPU = True
except Exception:
    xp = _np
    _GPU = False


def gpu_available() -> bool:
    return _GPU


def to_numpy(a):
    """Device -> host (no-op on the NumPy backend)."""
    if _GPU:
        import cupy
        return cupy.asnumpy(a)
    return _np.asarray(a)


def asarray(a, dtype=None):
    return xp.asarray(a, dtype=dtype)
