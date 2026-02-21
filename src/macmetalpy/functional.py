"""Functional programming tools -- vectorize, apply_along_axis, apply_over_axes.

These run on CPU (transferring data from GPU) since they apply arbitrary Python
functions. Results are returned as macmetalpy GPU arrays.
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


def vectorize(pyfunc, otypes=None, excluded=None, signature=None):
    """Vectorize a scalar Python function, returning macmetalpy arrays.

    Wraps numpy.vectorize but converts inputs from GPU to CPU and wraps
    outputs back as macmetalpy arrays.
    """
    np_vfunc = np.vectorize(pyfunc, otypes=otypes, excluded=excluded,
                            signature=signature)

    def wrapper(*args, **kwargs):
        np_args = [_to_numpy(a) for a in args]
        np_kwargs = {k: _to_numpy(v) for k, v in kwargs.items()}
        result = np_vfunc(*np_args, **np_kwargs)
        if isinstance(result, tuple):
            return tuple(creation.array(r) for r in result)
        return creation.array(result)

    wrapper.__name__ = getattr(pyfunc, '__name__', 'vectorized')
    wrapper.__doc__ = pyfunc.__doc__
    return wrapper


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Apply a function along a given axis of an array.

    Transfers data to CPU, applies via numpy, returns as GPU array.
    """
    arr_np = _to_numpy(arr)
    result = np.apply_along_axis(func1d, axis, arr_np, *args, **kwargs)
    return creation.array(result)


def apply_over_axes(func, a, axes):
    """Apply a function repeatedly over multiple axes.

    Transfers data to CPU, applies via numpy, returns as GPU array.
    """
    a_np = _to_numpy(a)
    result = np.apply_over_axes(func, a_np, axes)
    return creation.array(result)
