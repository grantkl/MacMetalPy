"""Sorting functions (CuPy-compatible)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def sort(a, axis=-1):
    """Return a sorted copy of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.sort(a.get(), axis=axis)
    return creation.array(result)


def argsort(a, axis=-1):
    """Return the indices that would sort an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.argsort(a.get(), axis=axis)
    return creation.array(result)


def unique(ar, return_index=False, return_inverse=False, return_counts=False,
           axis=None, equal_nan=True):
    """Find the unique elements of an array.

    Parameters
    ----------
    ar : array_like
        Input array.
    return_index : bool, optional
        If True, return indices of first occurrences.
    return_inverse : bool, optional
        If True, return indices to reconstruct input from unique.
    return_counts : bool, optional
        If True, return counts of each unique value.
    axis : int or None, optional
        Axis along which to find unique values.
    equal_nan : bool, optional
        If True (default), treat NaN values as equal.
    """
    if not isinstance(ar, ndarray):
        ar = creation.asarray(ar)
    np_result = np.unique(
        ar.get(),
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        equal_nan=equal_nan,
    )
    if isinstance(np_result, tuple):
        return tuple(creation.array(r) for r in np_result)
    return creation.array(np_result)


def searchsorted(a, v, side='left'):
    """Find indices where elements should be inserted to maintain order."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    v_np = v.get() if isinstance(v, ndarray) else np.asarray(v)
    result = np.searchsorted(a.get(), v_np, side=side)
    return creation.array(np.asarray(result))


def lexsort(keys):
    """Perform indirect stable sort using sequence of keys."""
    np_keys = [k.get() if isinstance(k, ndarray) else np.asarray(k) for k in keys]
    result = np.lexsort(np_keys)
    return creation.array(result)


def partition(a, kth, axis=-1):
    """Return a partitioned copy of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.partition(a.get(), kth, axis=axis)
    return creation.array(result)


def argpartition(a, kth, axis=-1):
    """Return indices that would partition an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.argpartition(a.get(), kth, axis=axis)
    return creation.array(result)


def msort(a):
    """Return a sorted copy along the first axis."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.sort(a.get(), axis=0)
    return creation.array(result)


def sort_complex(a):
    """Sort a complex array by real then imaginary part.

    Note: Metal does not support complex dtypes, so the result is
    converted to float32 (real part only) when the input is non-complex,
    matching the sorted order that ``np.sort_complex`` would produce.
    """
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_data = a.get()
    result = np.sort_complex(np_data)
    # Metal has no complex dtype support; fall back to float32 for real inputs.
    if not np.issubdtype(result.dtype, np.complexfloating):
        result = result.real.astype(np.float32)
    else:
        # For genuinely complex input, store as float32 pairs is not feasible.
        # Return sorted real parts as float32.
        result = result.real.astype(np.float32)
    return creation.array(result)
