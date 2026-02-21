"""Complex number operations (NumPy-compatible API)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def angle(z, deg=False):
    """Return the angle of the complex argument."""
    if not isinstance(z, ndarray):
        z = creation.asarray(z)
    result_np = np.angle(z.get(), deg=deg)
    return creation.array(result_np)


def real(val):
    """Return the real part of the complex argument."""
    if not isinstance(val, ndarray):
        val = creation.asarray(val)
    result_np = np.real(val.get())
    return creation.array(np.ascontiguousarray(result_np))


def imag(val):
    """Return the imaginary part of the complex argument."""
    if not isinstance(val, ndarray):
        val = creation.asarray(val)
    result_np = np.imag(val.get())
    return creation.array(np.ascontiguousarray(result_np))


def conj(x):
    """Return the complex conjugate, element-wise."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    result_np = np.conj(x.get())
    return creation.array(result_np)


conjugate = conj


def real_if_close(a, tol=100):
    """If input is complex with all imaginary parts close to zero, return real parts."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result_np = np.real_if_close(a.get(), tol=tol)
    return creation.array(np.ascontiguousarray(result_np))
