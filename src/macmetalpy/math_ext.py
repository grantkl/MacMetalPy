"""Extended math functions (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import ndarray
from . import creation


def _ensure(x):
    return x if isinstance(x, ndarray) else creation.asarray(x)


def sinc(x):
    """Return the sinc function."""
    x = _ensure(x)
    return creation.array(np.sinc(x.get()))


def i0(x):
    """Modified Bessel function of the first kind, order 0."""
    x = _ensure(x)
    result = np.i0(x.get())
    return creation.array(np.asarray(result))


def convolve(a, v, mode='full'):
    """Returns the discrete, linear convolution of two one-dimensional sequences."""
    a = _ensure(a)
    v = _ensure(v)
    return creation.array(np.convolve(a.get(), v.get(), mode=mode))


def interp(x, xp, fp, left=None, right=None):
    """One-dimensional linear interpolation."""
    x = _ensure(x)
    xp = _ensure(xp)
    fp = _ensure(fp)
    result = np.interp(x.get(), xp.get(), fp.get(), left=left, right=right)
    return creation.array(np.asarray(result))


def fix(x):
    """Round to nearest integer towards zero."""
    x = _ensure(x)
    return x._unary_op("trunc_op")


def unwrap(p, discont=None, axis=-1):
    """Unwrap by changing deltas between values to 2*pi complement."""
    p = _ensure(p)
    result = np.unwrap(p.get(), discont=discont, axis=axis)
    return creation.array(result)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Integrate along the given axis using the composite trapezoidal rule."""
    y = _ensure(y)
    x_np = x.get() if isinstance(x, ndarray) else x
    _trap = getattr(np, 'trapezoid', np.trapz)
    result = _trap(y.get(), x=x_np, dx=dx, axis=axis)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)
