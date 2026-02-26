"""Complex number operations (NumPy-compatible API)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray, _c_contiguous_strides, _wrap_np
from . import creation


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


def angle(z, deg=False):
    """Return the angle of the complex argument."""
    if not isinstance(z, ndarray):
        z = creation.asarray(z)
    return _wrap_np(np.angle(_get_np(z), deg=deg))


def real(val):
    """Return the real part of the complex argument."""
    if not isinstance(val, ndarray):
        val = creation.asarray(val)
    if not np.issubdtype(val.dtype, np.complexfloating):
        return val
    return _wrap_np(np.real(_get_np(val)))


def imag(val):
    """Return the imaginary part of the complex argument."""
    if not isinstance(val, ndarray):
        val = creation.asarray(val)
    return _wrap_np(np.imag(_get_np(val)))


def conj(x, **kwargs):
    """Return the complex conjugate, element-wise."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    if not np.issubdtype(x.dtype, np.complexfloating):
        return x
    return _wrap_np(np.conj(_get_np(x)))


conjugate = conj


def real_if_close(a, tol=100):
    """If input is complex with all imaginary parts close to zero, return real parts."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return _wrap_np(np.real_if_close(_get_np(a), tol=tol))
