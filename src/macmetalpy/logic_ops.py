"""Logic and comparison functions (CuPy-compatible, GPU-native)."""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def _ensure(x):
    return x if isinstance(x, ndarray) else creation.asarray(x)


# -- Logical functions (GPU-native via boolean shader) ---------------------


def _to_bool(x):
    """Convert to bool dtype so truthiness is correct for non-boolean inputs."""
    return x.astype(np.bool_) if x.dtype != np.bool_ else x


def logical_and(x1, x2):
    x1, x2 = _to_bool(_ensure(x1)), _to_bool(_ensure(x2))
    return x1._boolean_op(x2, "and_op")


def logical_or(x1, x2):
    x1, x2 = _to_bool(_ensure(x1)), _to_bool(_ensure(x2))
    return x1._boolean_op(x2, "or_op")


def logical_not(x):
    x = _to_bool(_ensure(x))
    return ~x


def logical_xor(x1, x2):
    x1, x2 = _to_bool(_ensure(x1)), _to_bool(_ensure(x2))
    return x1._boolean_op(x2, "xor_op")


# -- Comparison functions (GPU-native via comparison shader) ---------------


def greater(x1, x2):
    x1, x2 = _ensure(x1), _ensure(x2)
    return x1._comparison_op(x2, "gt_op")


def greater_equal(x1, x2):
    x1, x2 = _ensure(x1), _ensure(x2)
    return x1._comparison_op(x2, "ge_op")


def less(x1, x2):
    x1, x2 = _ensure(x1), _ensure(x2)
    return x1._comparison_op(x2, "lt_op")


def less_equal(x1, x2):
    x1, x2 = _ensure(x1), _ensure(x2)
    return x1._comparison_op(x2, "le_op")


def equal(x1, x2):
    x1, x2 = _ensure(x1), _ensure(x2)
    return x1._comparison_op(x2, "eq_op")


def not_equal(x1, x2):
    x1, x2 = _ensure(x1), _ensure(x2)
    return x1._comparison_op(x2, "ne_op")


# -- Type / value checks (CPU — diagnostic ops, not hot-path) -------------


def isneginf(x):
    x = _ensure(x)
    return creation.array(np.isneginf(x.get()))


def isposinf(x):
    x = _ensure(x)
    return creation.array(np.isposinf(x.get()))


def iscomplex(x):
    x = _ensure(x)
    return creation.array(np.iscomplex(x.get()))


def isreal(x):
    x = _ensure(x)
    return creation.array(np.isreal(x.get()))


def isscalar(element):
    return bool(np.isscalar(element.get() if isinstance(element, ndarray) else element))


def array_equiv(a1, a2):
    a1_np = a1.get() if isinstance(a1, ndarray) else a1
    a2_np = a2.get() if isinstance(a2, ndarray) else a2
    return bool(np.array_equiv(a1_np, a2_np))
