"""Extended math functions (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import ndarray, _c_contiguous_strides, _wrap_np
from . import creation


def _ensure(x):
    return x if isinstance(x, ndarray) else creation.asarray(x)


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


def sinc(x):
    """Return the sinc function."""
    x = _ensure(x)
    return _wrap_np(np.sinc(_get_np(x)))


def i0(x):
    """Modified Bessel function of the first kind, order 0."""
    x = _ensure(x)
    result = np.i0(_get_np(x))
    return ndarray._from_np_direct(np.asarray(result))


def convolve(a, v, mode='full'):
    """Returns the discrete, linear convolution of two one-dimensional sequences."""
    a = _ensure(a)
    v = _ensure(v)
    return _wrap_np(np.convolve(_get_np(a), _get_np(v), mode=mode))


def interp(x, xp, fp, left=None, right=None, period=None):
    """One-dimensional linear interpolation."""
    x = _ensure(x)
    xp = _ensure(xp)
    fp = _ensure(fp)
    kwargs = dict(left=left, right=right)
    if period is not None:
        kwargs['period'] = period
    result = np.interp(_get_np(x), _get_np(xp), _get_np(fp), **kwargs)
    return ndarray._from_np_direct(np.asarray(result))


def fix(x, out=None):
    """Round to nearest integer towards zero."""
    x = _ensure(x)
    # CPU fast path — avoid _unary_op dispatch overhead
    if x._np_data is not None and x.size < 4194304:
        result = _wrap_np(np.fix(x._np_data))
    else:
        result = x._unary_op("trunc_op")
    if out is not None:
        out._adopt_buffer(result.astype(out.dtype)._ensure_contiguous()._buffer)
        return out
    return result


def unwrap(p, discont=None, axis=-1, period=None):
    """Unwrap by changing deltas between values to 2*pi complement."""
    p = _ensure(p)
    kwargs = dict(discont=discont, axis=axis)
    if period is not None:
        kwargs['period'] = period
    return ndarray._from_np_direct(np.unwrap(_get_np(p), **kwargs))


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Integrate along the given axis using the composite trapezoidal rule."""
    y = _ensure(y)
    x_np = _get_np(x) if isinstance(x, ndarray) else x
    result = np.trapezoid(_get_np(y), x=x_np, dx=dx, axis=axis)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return ndarray._from_np_direct(result)


def piecewise(x, condlist, funclist, *args, **kw):
    """Evaluate a piecewise-defined function."""
    x = _ensure(x)
    np_condlist = [_get_np(c) if isinstance(c, ndarray) else c for c in condlist]
    return _wrap_np(np.piecewise(_get_np(x), np_condlist, funclist, *args, **kw))


def spacing(x):
    """Return the distance between x and the nearest adjacent number."""
    x = _ensure(x)
    return _wrap_np(np.spacing(_get_np(x)))


def isnat(x):
    """Test element-wise for NaT (not a time) and return result as a boolean array.

    Accepts numpy arrays directly since datetime64 is not a Metal-supported dtype.
    """
    if isinstance(x, ndarray):
        x_np = _get_np(x)
    else:
        x_np = np.asarray(x)
    return _wrap_np(np.isnat(x_np))
