"""Set operations (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import ndarray, _c_contiguous_strides
from . import creation


def _wrap_np(np_data):
    """Fast inline ndarray construction for known-good numpy arrays."""
    arr = ndarray.__new__(ndarray)
    arr._buffer = None
    arr._np_data = np_data
    arr._shape = np_data.shape
    arr._dtype = np_data.dtype
    arr._strides = _c_contiguous_strides(np_data.shape)
    arr._offset = 0
    arr._base = None
    return arr


def _cpu_view(a):
    """Get zero-copy numpy view (syncs if GPU-resident)."""
    if a._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
    return a._get_view()


def _get_np(a):
    """Get numpy data, preferring _np_data for CPU-resident arrays."""
    np_data = a._np_data
    if np_data is not None:
        return np_data
    return _cpu_view(a)


def _to_ndarray(np_result):
    """Convert a numpy result to ndarray, handling empty arrays."""
    if np_result.size == 0:
        return creation.empty(np_result.shape, dtype=np_result.dtype)
    return _wrap_np(np_result)


def union1d(ar1, ar2):
    """Find the union of two arrays."""
    if not isinstance(ar1, ndarray): ar1 = creation.asarray(ar1)
    if not isinstance(ar2, ndarray): ar2 = creation.asarray(ar2)
    return _to_ndarray(np.union1d(_get_np(ar1), _get_np(ar2)))


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """Find the intersection of two arrays."""
    if not isinstance(ar1, ndarray): ar1 = creation.asarray(ar1)
    if not isinstance(ar2, ndarray): ar2 = creation.asarray(ar2)
    result = np.intersect1d(_get_np(ar1), _get_np(ar2), assume_unique=assume_unique, return_indices=return_indices)
    if return_indices:
        return _to_ndarray(result[0]), _to_ndarray(result[1]), _to_ndarray(result[2])
    return _to_ndarray(result)


def setdiff1d(ar1, ar2, assume_unique=False):
    """Find set difference of two arrays."""
    if not isinstance(ar1, ndarray): ar1 = creation.asarray(ar1)
    if not isinstance(ar2, ndarray): ar2 = creation.asarray(ar2)
    return _to_ndarray(np.setdiff1d(_get_np(ar1), _get_np(ar2), assume_unique=assume_unique))


def setxor1d(ar1, ar2, assume_unique=False):
    """Find set exclusive-or of two arrays."""
    if not isinstance(ar1, ndarray): ar1 = creation.asarray(ar1)
    if not isinstance(ar2, ndarray): ar2 = creation.asarray(ar2)
    return _to_ndarray(np.setxor1d(_get_np(ar1), _get_np(ar2), assume_unique=assume_unique))


def in1d(ar1, ar2, assume_unique=False, invert=False, kind=None):
    """Test whether each element is also present in a second array.

    .. deprecated:: NumPy 2.0
        ``np.in1d`` was removed in NumPy 2.0.  This wrapper delegates to
        :func:`numpy.isin` which is the official replacement.  The function
        name ``in1d`` is kept for backward compatibility with CuPy-style code.
    """
    if not isinstance(ar1, ndarray): ar1 = creation.asarray(ar1)
    if not isinstance(ar2, ndarray): ar2 = creation.asarray(ar2)
    kwargs = dict(assume_unique=assume_unique, invert=invert)
    if kind is not None:
        kwargs["kind"] = kind
    return _wrap_np(np.isin(_get_np(ar1), _get_np(ar2), **kwargs))


def isin(element, test_elements, assume_unique=False, invert=False, kind=None):
    """Check if elements are present in test_elements."""
    if not isinstance(element, ndarray): element = creation.asarray(element)
    te = _get_np(test_elements) if isinstance(test_elements, ndarray) else np.asarray(test_elements)
    kwargs = dict(assume_unique=assume_unique, invert=invert)
    if kind is not None:
        kwargs["kind"] = kind
    return _wrap_np(np.isin(_get_np(element), te, **kwargs))
