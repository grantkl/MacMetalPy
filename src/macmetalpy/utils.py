"""Utility functions: frompyfunc, require, iterable, memory sharing, etc."""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def _to_numpy(a):
    """Convert macmetalpy ndarray to numpy array for delegation."""
    if isinstance(a, ndarray):
        return a.get()
    return a


def frompyfunc(func, nin, nout):
    """Create a generalized ufunc from an arbitrary Python function.

    Wraps numpy.frompyfunc but handles macmetalpy arrays.
    """
    np_ufunc = np.frompyfunc(func, nin, nout)

    def wrapper(*args, **kwargs):
        np_args = [_to_numpy(a) for a in args]
        np_kwargs = {k: _to_numpy(v) for k, v in kwargs.items()}
        result = np_ufunc(*np_args, **np_kwargs)
        if isinstance(result, tuple):
            return tuple(creation.array(np.asarray(r, dtype=np.float64)) for r in result)
        return creation.array(np.asarray(result, dtype=np.float64))

    return wrapper


def require(a, dtype=None, requirements=None):
    """Return an ndarray of the given type and target requirements."""
    np_a = _to_numpy(a)
    result = np.require(np_a, dtype=dtype, requirements=requirements)
    return creation.array(result)


def iterable(y):
    """Check whether or not an object can be iterated over."""
    if isinstance(y, ndarray):
        return y.ndim > 0
    return np.iterable(y)


def _get_root_buffer(a):
    """Walk the base chain to get the root buffer of an ndarray."""
    if not isinstance(a, ndarray):
        return None
    root = a
    while root._base is not None:
        root = root._base
    return root._buffer


def may_share_memory(a, b, max_work=None):
    """Determine if two arrays might share memory."""
    if isinstance(a, ndarray) and isinstance(b, ndarray):
        return _get_root_buffer(a) is _get_root_buffer(b)
    return False


def shares_memory(a, b, max_work=None):
    """Determine if two arrays share memory."""
    if isinstance(a, ndarray) and isinstance(b, ndarray):
        return _get_root_buffer(a) is _get_root_buffer(b)
    return False


def isdtype(dtype, kind):
    """Determine if a provided dtype is of a specified data type kind."""
    return np.isdtype(dtype, kind)


def info(object=None, maxwidth=76, output=None, toplevel='numpy'):
    """Get help information for an array, dtype, or function."""
    np.info(object, maxwidth=maxwidth, output=output, toplevel=toplevel)
