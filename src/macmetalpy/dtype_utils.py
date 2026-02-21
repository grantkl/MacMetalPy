"""Dtype inspection utilities, type aliases, and constants.

Thin wrappers around NumPy dtype functions that also accept macmetalpy ndarrays.
"""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray

# ── Constants ─────────────────────────────────────────────────────────
euler_gamma = 0.5772156649015329

# ── Dtype aliases ─────────────────────────────────────────────────────
complex64 = np.complex64
int_ = np.int64
float_ = np.float64
complex_ = np.complex128
intp = np.intp
uintp = np.uintp


# ── Helpers ───────────────────────────────────────────────────────────

def _extract_dtype(x):
    """Extract dtype from macmetalpy ndarray or numpy array, or return as-is."""
    if isinstance(x, ndarray):
        return x.dtype
    return x


def _extract_numpy(x):
    """Convert macmetalpy ndarray to numpy array for delegation."""
    if isinstance(x, ndarray):
        return x.get()
    return x


# ── Dtype inspection functions ────────────────────────────────────────

def can_cast(from_, to, casting='safe'):
    """Check if a cast between data types can occur according to the casting rule."""
    from_ = _extract_dtype(from_)
    return bool(np.can_cast(from_, to, casting=casting))


def promote_types(type1, type2):
    """Return the dtype with the smallest size and kind to which both types can be safely cast."""
    return np.promote_types(type1, type2)


def result_type(*arrays_and_dtypes):
    """Return the type that results from applying numpy type promotion rules."""
    converted = []
    for a in arrays_and_dtypes:
        if isinstance(a, ndarray):
            converted.append(a.get())
        else:
            converted.append(a)
    return np.result_type(*converted)


def common_type(*arrays):
    """Return a scalar type common to the input arrays."""
    converted = [a.get() if isinstance(a, ndarray) else a for a in arrays]
    return np.common_type(*converted)


def min_scalar_type(a):
    """Return the minimum scalar type for a value."""
    a = _extract_numpy(a)
    return np.min_scalar_type(a)


def finfo(dtype):
    """Return machine limits for floating point types."""
    return np.finfo(dtype)


def iinfo(dtype):
    """Return machine limits for integer types."""
    return np.iinfo(dtype)


def issubdtype(arg1, arg2):
    """Return True if first argument is a typecode lower/equal in type hierarchy."""
    return bool(np.issubdtype(arg1, arg2))


# ── Utility functions ─────────────────────────────────────────────────

def ndim(a):
    """Return the number of dimensions of an array."""
    if isinstance(a, ndarray):
        return a.ndim
    return np.ndim(a)


def shape(a):
    """Return the shape of an array."""
    if isinstance(a, ndarray):
        return a.shape
    return np.shape(a)


def size(a, axis=None):
    """Return the number of elements along a given axis."""
    if isinstance(a, ndarray):
        if axis is None:
            return a.size
        return a.shape[axis]
    return np.size(a, axis=axis)
