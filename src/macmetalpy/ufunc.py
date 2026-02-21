"""Ufunc-like objects for element-wise and accumulate operations."""

from __future__ import annotations

import numpy as np

__all__ = ["maximum", "minimum"]


class _Ufunc2:
    """Lightweight binary ufunc supporting __call__ and accumulate.

    For max/min the __call__ path uses a native Metal GPU kernel;
    accumulate still falls back to CPU (scan is inherently sequential).
    """

    def __init__(self, np_func, gpu_op_name: str):
        self._np_func = np_func
        self._gpu_op = gpu_op_name

    def __call__(self, a, b):
        """Element-wise operation via Metal GPU kernel."""
        from .ndarray import ndarray
        from . import creation

        if not isinstance(a, ndarray):
            a = creation.asarray(a)
        if not isinstance(b, ndarray):
            b = creation.asarray(b)

        return a._binary_op(b, self._gpu_op)

    def accumulate(self, a, axis=0):
        """Cumulative operation via CPU fallback."""
        from .ndarray import ndarray
        from . import creation

        if not isinstance(a, ndarray):
            a = creation.asarray(a)

        result_np = self._np_func.accumulate(a.get(), axis=axis)
        return creation.array(result_np)


maximum = _Ufunc2(np.maximum, "max_op")
minimum = _Ufunc2(np.minimum, "min_op")
