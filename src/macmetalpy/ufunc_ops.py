"""Math ufunc operations (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import ndarray
from . import creation


def _ensure(x):
    if not isinstance(x, ndarray):
        return creation.asarray(x)
    return x


def _ensure2(a, b):
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    return a, b


# ---- binary arithmetic ufuncs ----

def add(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "add_op")


def subtract(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "sub_op")


def multiply(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "mul_op")


def divide(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    # True division always returns float (match NumPy)
    if np.issubdtype(x1.dtype, np.integer):
        x1 = creation.array(x1.get().astype(np.float32))
    if np.issubdtype(x2.dtype, np.integer):
        x2 = creation.array(x2.get().astype(np.float32))
    return x1._binary_op(x2, "div_op")


def true_divide(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    if np.issubdtype(x1.dtype, np.integer):
        x1 = creation.array(x1.get().astype(np.float32))
    if np.issubdtype(x2.dtype, np.integer):
        x2 = creation.array(x2.get().astype(np.float32))
    return x1._binary_op(x2, "div_op")


def floor_divide(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "floor_divide_op")


def float_power(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1.astype(np.float32)._binary_op(x2.astype(np.float32), "pow_op")


def fmod(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "fmod_op")


# ---- unary math ufuncs ----

def exp2(x):
    x = _ensure(x)
    return x._unary_op("exp2_op")


def expm1(x):
    x = _ensure(x)
    return x._unary_op("expm1_op")


def log1p(x):
    x = _ensure(x)
    return x._unary_op("log1p_op")


def cbrt(x):
    x = _ensure(x)
    return x._unary_op("cbrt_op")


def reciprocal(x):
    x = _ensure(x)
    return x._unary_op("reciprocal_op")


def rint(x):
    x = _ensure(x)
    return x._unary_op("rint_op")


def trunc(x):
    x = _ensure(x)
    return x._unary_op("trunc_op")


def absolute(x):
    x = _ensure(x)
    return x._unary_op("abs_op")


def fabs(x):
    x = _ensure(x)
    return x._unary_op("abs_op")


def positive(x):
    x = _ensure(x)
    return x._unary_op("positive_op")


# ---- two-argument math ufuncs ----

def arctan2(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "atan2_op")


def hypot(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "hypot_op")


def logaddexp(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "logaddexp_op")


def logaddexp2(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "logaddexp2_op")


def heaviside(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "heaviside_op")


def copysign(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "copysign_op")


def nextafter(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "nextafter_op")


# ---- inverse hyperbolic ----

def arcsinh(x):
    x = _ensure(x)
    return x._unary_op("asinh_op")


def arccosh(x):
    x = _ensure(x)
    return x._unary_op("acosh_op")


def arctanh(x):
    x = _ensure(x)
    return x._unary_op("atanh_op")


# ---- angle conversions ----

def degrees(x):
    x = _ensure(x)
    return x._unary_op("degrees_op")


def radians(x):
    x = _ensure(x)
    return x._unary_op("radians_op")


def deg2rad(x):
    x = _ensure(x)
    return x._unary_op("radians_op")


def rad2deg(x):
    x = _ensure(x)
    return x._unary_op("degrees_op")


# ---- special ufuncs ----

def signbit(x):
    x = _ensure(x)
    return x._unary_predicate_op("signbit_op")


def modf(x):
    x = _ensure(x)
    frac, intg = np.modf(x.get())
    return creation.array(frac), creation.array(intg)


def ldexp(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return creation.array(np.ldexp(x1.get(), x2.get()))


def frexp(x):
    x = _ensure(x)
    mant, exp_ = np.frexp(x.get())
    return creation.array(mant), creation.array(exp_)


# ---- amax/amin and fmax/fmin ----

def amax(a, axis=None):
    a = _ensure(a)
    if axis is None:
        return a._reduce("reduce_max")
    return creation.array(np.amax(a.get(), axis=axis))


def amin(a, axis=None):
    a = _ensure(a)
    if axis is None:
        return a._reduce("reduce_min")
    return creation.array(np.amin(a.get(), axis=axis))


def fmax(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "fmax_op")


def fmin(x1, x2):
    x1, x2 = _ensure2(x1, x2)
    return x1._binary_op(x2, "fmin_op")
