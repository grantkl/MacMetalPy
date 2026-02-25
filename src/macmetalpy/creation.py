"""Array creation functions (CuPy-compatible API)."""

from __future__ import annotations

from typing import Union

import numpy as np

from ._dtypes import resolve_dtype
from ._metal_backend import MetalBackend
from .ndarray import ndarray, _c_contiguous_strides

__all__ = [
    "empty",
    "zeros",
    "ones",
    "full",
    "arange",
    "array",
    "asarray",
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
    "linspace",
    "eye",
]


def _wrap_np(np_data):
    """Inline ndarray construction for creation ops (known-good dtype, C-contiguous)."""
    arr = ndarray.__new__(ndarray)
    arr._buffer = None
    arr._np_data = np_data
    arr._shape = np_data.shape
    arr._dtype = np_data.dtype
    arr._strides = _c_contiguous_strides(np_data.shape)
    arr._offset = 0
    arr._base = None
    return arr


def _validate_device(device):
    """Validate the device parameter (only None and 'cpu' are supported)."""
    if device is not None and device != "cpu":
        raise ValueError(f"Unsupported device: {device!r}")


def empty(shape, dtype=None, order='C', *, like=None, device=None) -> ndarray:
    """Return a new array of given shape and type, without initialising entries."""
    _validate_device(device)
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    return _wrap_np(np.empty(shape, dtype=resolve_dtype(dtype)))


def zeros(shape, dtype=None, order='C', *, like=None, device=None) -> ndarray:
    """Return a new array of given shape filled with zeros."""
    _validate_device(device)
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    return _wrap_np(np.zeros(shape, dtype=resolve_dtype(dtype)))


def ones(shape, dtype=None, order='C', *, like=None, device=None) -> ndarray:
    """Return a new array of given shape filled with ones."""
    _validate_device(device)
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    return _wrap_np(np.ones(shape, dtype=resolve_dtype(dtype)))


def full(shape, fill_value, dtype=None, order='C', *, like=None, device=None) -> ndarray:
    """Return a new array of given shape filled with *fill_value*."""
    _validate_device(device)
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    return _wrap_np(np.full(shape, fill_value, dtype=resolve_dtype(dtype)))


def arange(start_or_stop, stop=None, step=1, dtype=None, *, like=None, device=None) -> ndarray:
    """Return evenly spaced values within a given interval."""
    _validate_device(device)
    if stop is None:
        start, stop = 0, start_or_stop
    else:
        start = start_or_stop
    if dtype is not None:
        dtype = resolve_dtype(dtype)
    else:
        # Resolve dtype from arguments to avoid float64 intermediate copy
        probe = np.result_type(type(start), type(stop), type(step))
        dtype = resolve_dtype(probe)
    return _wrap_np(np.arange(start, stop, step, dtype=dtype))


def array(obj, dtype=None) -> ndarray:
    """Create an ndarray from a list, NumPy array, or macmetalpy ndarray."""
    if isinstance(obj, ndarray):
        np_data = obj._np_data if obj._np_data is not None else obj.get()
    else:
        np_data = np.asarray(obj)

    if dtype is not None:
        dtype = resolve_dtype(dtype)
        np_data = np_data.astype(dtype, copy=False)
    else:
        from ._dtypes import _PASSTHROUGH_DTYPES
        if np_data.dtype in _PASSTHROUGH_DTYPES:
            return _wrap_np(np_data)
    return ndarray._from_np_direct(np_data)


def asarray(obj, dtype=None) -> ndarray:
    """Convert input to an ndarray; no-copy if already the right type/dtype."""
    if isinstance(obj, ndarray):
        if dtype is None or np.dtype(dtype) == obj.dtype:
            return obj
    return array(obj, dtype=dtype)


def zeros_like(a, dtype=None, order='K', *, device=None) -> ndarray:
    """Return an array of zeros with the same shape as *a*."""
    _validate_device(device)
    if isinstance(a, ndarray):
        d = dtype if dtype is not None else a.dtype
        return _wrap_np(np.zeros(a.shape, dtype=d))
    a = np.asarray(a)
    return _wrap_np(np.zeros_like(a, dtype=dtype))


def ones_like(a, dtype=None, order='K', *, device=None) -> ndarray:
    """Return an array of ones with the same shape as *a*."""
    _validate_device(device)
    if isinstance(a, ndarray):
        d = dtype if dtype is not None else a.dtype
        return _wrap_np(np.ones(a.shape, dtype=d))
    a = np.asarray(a)
    return _wrap_np(np.ones_like(a, dtype=dtype))


def empty_like(a, dtype=None, order='K', *, device=None) -> ndarray:
    """Return an uninitialised array with the same shape as *a*."""
    _validate_device(device)
    if isinstance(a, ndarray):
        d = dtype if dtype is not None else a.dtype
        return _wrap_np(np.empty(a.shape, dtype=d))
    a = np.asarray(a)
    return _wrap_np(np.empty_like(a, dtype=dtype))


def full_like(a, fill_value, dtype=None, order='K', *, device=None) -> ndarray:
    """Return a filled array with the same shape as *a*."""
    _validate_device(device)
    if isinstance(a, ndarray):
        shape = a.shape
        dtype = dtype if dtype is not None else a.dtype
    else:
        a = np.asarray(a)
        shape = a.shape
    return full(shape, fill_value, dtype=dtype)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, like=None, device=None) -> ndarray:
    """Return evenly spaced numbers over a specified interval."""
    _validate_device(device)
    np_result = np.linspace(start, stop, num, endpoint=endpoint, retstep=retstep, axis=axis)
    if retstep:
        np_data, step = np_result
    else:
        np_data = np_result
    if dtype is not None:
        dtype = resolve_dtype(dtype)
        np_data = np_data.astype(dtype, copy=False)
    arr = ndarray._from_np_direct(np_data)
    if retstep:
        return arr, step
    return arr


def eye(N, M=None, k=0, dtype=None, order='C', *, like=None, device=None) -> ndarray:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
    _validate_device(device)
    if M is None:
        M = N
    return _wrap_np(np.eye(N, M, k=k, dtype=resolve_dtype(dtype)))


def diag(v, k=0):
    """Extract a diagonal or construct a diagonal array."""
    if isinstance(v, ndarray):
        v_np = v.get()
    else:
        v_np = np.asarray(v)
    result = np.diag(v_np, k=k)
    return ndarray._from_np_direct(result)


def identity(n, dtype=None, *, like=None):
    """Return the identity array (alias for eye)."""
    return eye(n, dtype=dtype)


def tri(N, M=None, k=0, dtype=None, *, like=None):
    """Return array with ones at and below the given diagonal."""
    return _wrap_np(np.tri(N, M, k, dtype=resolve_dtype(dtype)))


def triu(m, k=0):
    """Upper triangle of an array."""
    if isinstance(m, ndarray):
        m_np = m.get()
    else:
        m_np = np.asarray(m)
    return ndarray._from_np_direct(np.triu(m_np, k=k))


def tril(m, k=0):
    """Lower triangle of an array."""
    if isinstance(m, ndarray):
        m_np = m.get()
    else:
        m_np = np.asarray(m)
    return ndarray._from_np_direct(np.tril(m_np, k=k))


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """Return numbers spaced evenly on a log scale."""
    np_data = np.logspace(start, stop, num, endpoint=endpoint, base=base, axis=axis)
    if dtype is not None:
        np_data = np_data.astype(resolve_dtype(dtype), copy=False)
    return ndarray._from_np_direct(np_data)


def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    """Return coordinate matrices from coordinate vectors."""
    np_arrays = []
    for x in xi:
        if isinstance(x, ndarray):
            np_arrays.append(x._np_data if x._np_data is not None else x.get())
        else:
            np_arrays.append(np.asarray(x))
    results = np.meshgrid(*np_arrays, copy=copy, sparse=sparse, indexing=indexing)
    return [ndarray._from_np_direct(r) for r in results]


def indices(dimensions, dtype=int, sparse=False):
    """Return an array representing the indices of a grid."""
    result = np.indices(dimensions, dtype=dtype, sparse=sparse)
    if sparse:
        return [ndarray._from_np_direct(r) for r in result]
    return array(result)


def fromfunction(function, shape, dtype=float):
    """Construct array by executing function over each coordinate."""
    result = np.fromfunction(function, shape, dtype=dtype)
    return ndarray._from_np_direct(result)


def diagflat(v, k=0):
    """Create a diagonal array with the flattened input as diagonal."""
    if isinstance(v, ndarray):
        v_np = v.get()
    else:
        v_np = np.asarray(v)
    result = np.diagflat(v_np, k=k)
    return ndarray._from_np_direct(result)


def vander(x, N=None, increasing=False):
    """Generate a Vandermonde matrix."""
    if isinstance(x, ndarray):
        x_np = x.get()
    else:
        x_np = np.asarray(x)
    result = np.vander(x_np, N=N, increasing=increasing)
    return ndarray._from_np_direct(result)


def asanyarray(a, dtype=None):
    """Convert input to an ndarray (pass-through for ndarrays)."""
    return asarray(a, dtype=dtype)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """Return numbers spaced evenly on a log scale (a geometric progression)."""
    np_data = np.geomspace(start, stop, num, endpoint=endpoint, axis=axis)
    if dtype is not None:
        np_data = np_data.astype(resolve_dtype(dtype), copy=False)
    return ndarray._from_np_direct(np_data)


def frombuffer(buffer, dtype=float, count=-1, offset=0):
    """Interpret a buffer as a 1-D array."""
    dtype = resolve_dtype(dtype)
    np_data = np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
    return _wrap_np(np_data)


def asarray_chkfinite(a, dtype=None, order=None):
    """Convert input to an array, checking for NaN and Inf.

    Raises ValueError if the input contains NaN or Inf.
    """
    if isinstance(a, ndarray):
        a = a.get()
    np_data = np.asarray_chkfinite(a, dtype=dtype, order=order)
    return array(np_data)


def fromiter(iterable, dtype, count=-1):
    """Create a new 1-D array from an iterable object."""
    np_data = np.fromiter(iterable, dtype=dtype, count=count)
    return array(np_data)


def fromstring(string, dtype=float, count=-1, *, sep=''):
    """Create a new 1-D array from a string."""
    np_data = np.fromstring(string, dtype=dtype, count=count, sep=sep)
    return array(np_data)


def asfarray(a, dtype=np.float64):
    """Return an array converted to a float type.

    Deprecated in NumPy 1.26 but provided for API compatibility.
    """
    if isinstance(a, ndarray):
        a = a.get()
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float64
    np_data = np.asarray(a, dtype=dtype)
    return array(np_data)
