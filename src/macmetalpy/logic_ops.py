"""Logic and comparison functions (CuPy-compatible, GPU-native)."""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray, _c_contiguous_strides, _wrap_np, _GPU_THRESHOLD_MEMORY
from . import creation


def _ensure(x):
    return x if isinstance(x, ndarray) else creation.asarray(x)


def _cpu_view(x):
    """Get zero-copy numpy view (syncs if GPU-resident)."""
    if x._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
    return x._get_view()


# -- Logical functions (GPU-native via boolean shader) ---------------------


def _to_bool(x):
    """Convert to bool dtype so truthiness is correct for non-boolean inputs."""
    if x.dtype == np.bool_:
        return x
    return x.astype(np.bool_)


def _logical_binary(x1, x2, np_func, gpu_op):
    x1, x2 = _ensure(x1), _ensure(x2)
    if x1.size < _GPU_THRESHOLD_MEMORY and x2.size < _GPU_THRESHOLD_MEMORY:
        a = x1._np_data if x1._np_data is not None else _cpu_view(x1)
        b = x2._np_data if x2._np_data is not None else _cpu_view(x2)
        return _wrap_np(np_func(a, b))
    if x1.dtype != np.bool_:
        x1 = x1.astype(np.bool_)
    if x2.dtype != np.bool_:
        x2 = x2.astype(np.bool_)
    return x1._boolean_op(x2, gpu_op)


def logical_and(x1, x2, **kwargs):
    return _logical_binary(x1, x2, np.logical_and, "and_op")


def logical_or(x1, x2, **kwargs):
    return _logical_binary(x1, x2, np.logical_or, "or_op")


def logical_not(x, **kwargs):
    x = _ensure(x)
    if x.size < _GPU_THRESHOLD_MEMORY:
        a = x._np_data if x._np_data is not None else _cpu_view(x)
        return _wrap_np(np.logical_not(a))
    if x.dtype != np.bool_:
        x = x.astype(np.bool_)
    return ~x


def logical_xor(x1, x2, **kwargs):
    return _logical_binary(x1, x2, np.logical_xor, "xor_op")


# -- Comparison functions (GPU-native via comparison shader) ---------------


def greater(x1, x2, **kwargs):
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._comparison_op(x2, "gt_op")


def greater_equal(x1, x2, **kwargs):
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._comparison_op(x2, "ge_op")


def less(x1, x2, **kwargs):
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._comparison_op(x2, "lt_op")


def less_equal(x1, x2, **kwargs):
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._comparison_op(x2, "le_op")


def equal(x1, x2, **kwargs):
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._comparison_op(x2, "eq_op")


def not_equal(x1, x2, **kwargs):
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._comparison_op(x2, "ne_op")


# -- Type / value checks (CPU — diagnostic ops, not hot-path) -------------


def isneginf(x, out=None):
    x = _ensure(x)
    a = x._np_data if x._np_data is not None else _cpu_view(x)
    result = _wrap_np(np.isneginf(a))
    if out is not None:
        out.set(result.get())
        return out
    return result


def isposinf(x, out=None):
    x = _ensure(x)
    a = x._np_data if x._np_data is not None else _cpu_view(x)
    result = _wrap_np(np.isposinf(a))
    if out is not None:
        out.set(result.get())
        return out
    return result


def iscomplex(x):
    x = _ensure(x)
    a = x._np_data if x._np_data is not None else _cpu_view(x)
    return _wrap_np(np.asarray(np.iscomplex(a)))


def isreal(x):
    x = _ensure(x)
    a = x._np_data if x._np_data is not None else _cpu_view(x)
    return _wrap_np(np.asarray(np.isreal(a)))


def isscalar(element):
    return bool(np.isscalar(element.get() if isinstance(element, ndarray) else element))


def array_equiv(a1, a2):
    a1_np = a1.get() if isinstance(a1, ndarray) else a1
    a2_np = a2.get() if isinstance(a2, ndarray) else a2
    return bool(np.array_equiv(a1_np, a2_np))


# -- Object-level type checks (CPU — check dtype.kind, not element values) --


def iscomplexobj(x):
    """Return True if x has a complex dtype (checks dtype.kind == 'c')."""
    if isinstance(x, ndarray):
        return x.dtype.kind == 'c'
    return bool(np.iscomplexobj(x))


def isrealobj(x):
    """Return True if x does NOT have a complex dtype."""
    return not iscomplexobj(x)


def isfortran(a):
    """Return True if the array is Fortran-contiguous.

    macmetalpy arrays are always row-major (C-contiguous), so this always
    returns False.
    """
    return False
