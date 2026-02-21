"""Index expression objects -- c_, r_, s_, mgrid, ogrid.

These mirror numpy's index trick objects that use __getitem__ with slice objects.
Results are returned as macmetalpy GPU arrays (except s_ which returns plain
index tuples/slices).
"""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def _to_numpy(x):
    """Convert macmetalpy ndarray to numpy, pass through everything else."""
    if isinstance(x, ndarray):
        return x.get()
    return x


class _CClass:
    """Translates slice objects to concatenation along the second axis (column-wise).

    Usage: c_[array1, array2] or c_[1:5, 7:10]
    """

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        np_key = tuple(_to_numpy(k) for k in key)
        result = np.c_.__getitem__(np_key)
        return creation.array(result)


class _RClass:
    """Translates slice objects to concatenation along the first axis (row-wise).

    Usage: r_[array1, array2] or r_[1:5, '0,2', 7:10]
    """

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        np_key = tuple(_to_numpy(k) for k in key)
        result = np.r_.__getitem__(np_key)
        return creation.array(result)


class _IndexExpression:
    """Index expression helper -- returns index tuples/slices.

    Usage: s_[1:5] returns slice(1, 5)
    """

    def __getitem__(self, key):
        return key


class _MGridClass:
    """Return dense multi-dimensional meshgrid.

    Usage: mgrid[0:5, 0:5]
    """

    def __getitem__(self, key):
        result = np.mgrid.__getitem__(key)
        if isinstance(result, np.ndarray):
            return creation.array(result)
        # result is a list of arrays
        return [creation.array(r) for r in result]


class _OGridClass:
    """Return open multi-dimensional meshgrid.

    Usage: ogrid[0:5, 0:5]
    """

    def __getitem__(self, key):
        result = np.ogrid.__getitem__(key)
        if isinstance(result, np.ndarray):
            return creation.array(result)
        # result is a list of arrays
        return [creation.array(r) for r in result]


# Singleton instances
c_ = _CClass()
r_ = _RClass()
s_ = _IndexExpression()
mgrid = _MGridClass()
ogrid = _OGridClass()
