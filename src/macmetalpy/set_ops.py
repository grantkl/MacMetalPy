"""Set operations (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import ndarray
from . import creation


def _to_ndarray(np_result):
    """Convert a numpy result to ndarray, handling empty arrays."""
    if np_result.size == 0:
        return creation.empty(np_result.shape, dtype=np_result.dtype)
    return creation.array(np_result)


def union1d(ar1, ar2):
    """Find the union of two arrays."""
    if not isinstance(ar1, ndarray): ar1 = creation.asarray(ar1)
    if not isinstance(ar2, ndarray): ar2 = creation.asarray(ar2)
    return _to_ndarray(np.union1d(ar1.get(), ar2.get()))


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """Find the intersection of two arrays."""
    if not isinstance(ar1, ndarray): ar1 = creation.asarray(ar1)
    if not isinstance(ar2, ndarray): ar2 = creation.asarray(ar2)
    result = np.intersect1d(ar1.get(), ar2.get(), assume_unique=assume_unique, return_indices=return_indices)
    if return_indices:
        return _to_ndarray(result[0]), _to_ndarray(result[1]), _to_ndarray(result[2])
    return _to_ndarray(result)


def setdiff1d(ar1, ar2, assume_unique=False):
    """Find set difference of two arrays."""
    if not isinstance(ar1, ndarray): ar1 = creation.asarray(ar1)
    if not isinstance(ar2, ndarray): ar2 = creation.asarray(ar2)
    return _to_ndarray(np.setdiff1d(ar1.get(), ar2.get(), assume_unique=assume_unique))


def setxor1d(ar1, ar2, assume_unique=False):
    """Find set exclusive-or of two arrays."""
    if not isinstance(ar1, ndarray): ar1 = creation.asarray(ar1)
    if not isinstance(ar2, ndarray): ar2 = creation.asarray(ar2)
    return _to_ndarray(np.setxor1d(ar1.get(), ar2.get(), assume_unique=assume_unique))


def in1d(ar1, ar2, assume_unique=False, invert=False):
    """Test whether each element is also present in a second array."""
    if not isinstance(ar1, ndarray): ar1 = creation.asarray(ar1)
    if not isinstance(ar2, ndarray): ar2 = creation.asarray(ar2)
    return creation.array(np.in1d(ar1.get(), ar2.get(), assume_unique=assume_unique, invert=invert))


def isin(element, test_elements, assume_unique=False, invert=False):
    """Check if elements are present in test_elements."""
    if not isinstance(element, ndarray): element = creation.asarray(element)
    te = test_elements.get() if isinstance(test_elements, ndarray) else np.asarray(test_elements)
    return creation.array(np.isin(element.get(), te, assume_unique=assume_unique, invert=invert))
