"""Math ufunc operations (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import (ndarray, _fast_unary, _fast_binary,
    _UOP_EXP2, _UOP_EXPM1, _UOP_LOG1P, _UOP_CBRT, _UOP_RECIPROCAL,
    _UOP_RINT, _UOP_TRUNC, _UOP_ABS, _UOP_POSITIVE,
    _UOP_ASINH, _UOP_ACOSH, _UOP_ATANH, _UOP_DEGREES, _UOP_RADIANS,
    _OP_ADD, _OP_SUB, _OP_MUL, _OP_FLOOR_DIV,
    _OP_FMOD, _OP_ATAN2, _OP_HYPOT, _OP_LOGADDEXP, _OP_LOGADDEXP2,
    _OP_HEAVISIDE, _OP_COPYSIGN, _OP_NEXTAFTER, _OP_FMAX, _OP_FMIN)
from . import creation


def _ensure(x):
    if not isinstance(x, ndarray):
        return creation.asarray(x)
    return x


def _cpu_view(a):
    """Get zero-copy numpy view (syncs if GPU-resident)."""
    if a._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
    return a._get_view()


def _ensure2(a, b):
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    return a, b


# ---- binary arithmetic ufuncs ----

def add(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_ADD)
        if r is not None:
            return r
    if type(x1) is not ndarray:
        x1 = creation.asarray(x1)
    return x1._binary_op(x2, "add_op")


def subtract(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_SUB)
        if r is not None:
            return r
    if type(x1) is not ndarray:
        x1 = creation.asarray(x1)
    return x1._binary_op(x2, "sub_op")


def multiply(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_MUL)
        if r is not None:
            return r
    if type(x1) is not ndarray:
        x1 = creation.asarray(x1)
    return x1._binary_op(x2, "mul_op")


def divide(x1, x2):
    if type(x1) is not ndarray:
        x1 = creation.asarray(x1)
    if type(x2) is not ndarray:
        x2 = creation.asarray(x2)
    # True division always returns float (match NumPy)
    if np.issubdtype(x1.dtype, np.integer):
        x1 = x1.astype(np.float32)
    if np.issubdtype(x2.dtype, np.integer):
        x2 = x2.astype(np.float32)
    return x1._binary_op(x2, "div_op")


def true_divide(x1, x2):
    if type(x1) is not ndarray:
        x1 = creation.asarray(x1)
    if type(x2) is not ndarray:
        x2 = creation.asarray(x2)
    if np.issubdtype(x1.dtype, np.integer):
        x1 = x1.astype(np.float32)
    if np.issubdtype(x2.dtype, np.integer):
        x2 = x2.astype(np.float32)
    return x1._binary_op(x2, "div_op")


def floor_divide(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_FLOOR_DIV)
        if r is not None:
            return r
    if type(x1) is not ndarray:
        x1 = creation.asarray(x1)
    return x1._binary_op(x2, "floor_divide_op")


def float_power(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1.astype(np.float32)._binary_op(x2.astype(np.float32), "pow_op")


def fmod(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_FMOD)
        if r is not None:
            return r
    if type(x1) is not ndarray:
        x1 = creation.asarray(x1)
    return x1._binary_op(x2, "fmod_op")


# ---- unary math ufuncs ----

def exp2(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_EXP2)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("exp2_op")


def expm1(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_EXPM1)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("expm1_op")


def log1p(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_LOG1P)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("log1p_op")


def cbrt(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_CBRT)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("cbrt_op")


def reciprocal(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_RECIPROCAL)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("reciprocal_op")


def rint(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_RINT)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("rint_op")


def trunc(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_TRUNC)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("trunc_op")


def absolute(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_ABS)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("abs_op")


def fabs(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_ABS)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("abs_op")


def positive(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_POSITIVE)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("positive_op")


# ---- two-argument math ufuncs ----

def arctan2(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_ATAN2)
        if r is not None:
            return r
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._binary_op(x2, "atan2_op")


def hypot(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_HYPOT)
        if r is not None:
            return r
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._binary_op(x2, "hypot_op")


def logaddexp(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_LOGADDEXP)
        if r is not None:
            return r
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._binary_op(x2, "logaddexp_op")


def logaddexp2(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_LOGADDEXP2)
        if r is not None:
            return r
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._binary_op(x2, "logaddexp2_op")


def heaviside(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_HEAVISIDE)
        if r is not None:
            return r
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._binary_op(x2, "heaviside_op")


def copysign(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_COPYSIGN)
        if r is not None:
            return r
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._binary_op(x2, "copysign_op")


def nextafter(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_NEXTAFTER)
        if r is not None:
            return r
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._binary_op(x2, "nextafter_op")


# ---- inverse hyperbolic ----

def arcsinh(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_ASINH)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("asinh_op")


def arccosh(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_ACOSH)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("acosh_op")


def arctanh(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_ATANH)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("atanh_op")


# ---- angle conversions ----

def degrees(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_DEGREES)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("degrees_op")


def radians(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_RADIANS)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("radians_op")


def deg2rad(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_RADIANS)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("radians_op")


def rad2deg(x):
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_DEGREES)
        if r is not None:
            return r
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_op("degrees_op")


# ---- special ufuncs ----

def signbit(x):
    if type(x) is not ndarray: x = creation.asarray(x)
    return x._unary_predicate_op("signbit_op")


def modf(x):
    x = _ensure(x)
    a = x._np_data if x._np_data is not None else _cpu_view(x)
    frac, intg = np.modf(a)
    return ndarray._from_np_direct(frac), ndarray._from_np_direct(intg)


def ldexp(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    a = x1._np_data if x1._np_data is not None else _cpu_view(x1)
    b = x2._np_data if x2._np_data is not None else _cpu_view(x2)
    return ndarray._from_np_direct(np.ldexp(a, b))


def frexp(x):
    x = _ensure(x)
    a = x._np_data if x._np_data is not None else _cpu_view(x)
    mant, exp_ = np.frexp(a)
    return ndarray._from_np_direct(mant), ndarray._from_np_direct(exp_)


# ---- amax/amin and fmax/fmin ----

def amax(a, axis=None, out=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    a = _ensure(a)
    if where is not np._NoValue:
        from .reductions import _apply_where
        a = _apply_where(a, where, -np.inf)
    if axis is None and not keepdims and initial is np._NoValue:
        result = a._reduce("reduce_max")
    else:
        np_data = a._np_data if a._np_data is not None else _cpu_view(a)
        kwargs = dict(axis=axis, keepdims=keepdims)
        if initial is not np._NoValue:
            kwargs['initial'] = initial
        result = ndarray._from_np_direct(np.amax(np_data, **kwargs))
    if out is not None:
        from .reductions import _copy_to_out
        return _copy_to_out(result, out)
    return result


def amin(a, axis=None, out=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    a = _ensure(a)
    if where is not np._NoValue:
        from .reductions import _apply_where
        a = _apply_where(a, where, np.inf)
    if axis is None and not keepdims and initial is np._NoValue:
        result = a._reduce("reduce_min")
    else:
        np_data = a._np_data if a._np_data is not None else _cpu_view(a)
        kwargs = dict(axis=axis, keepdims=keepdims)
        if initial is not np._NoValue:
            kwargs['initial'] = initial
        result = ndarray._from_np_direct(np.amin(np_data, **kwargs))
    if out is not None:
        from .reductions import _copy_to_out
        return _copy_to_out(result, out)
    return result


def fmax(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_FMAX)
        if r is not None:
            return r
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._binary_op(x2, "fmax_op")


def fmin(x1, x2):
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_FMIN)
        if r is not None:
            return r
    if type(x1) is not ndarray: x1 = creation.asarray(x1)
    return x1._binary_op(x2, "fmin_op")
