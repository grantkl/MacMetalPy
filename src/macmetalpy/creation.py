"""Array creation functions (CuPy-compatible API)."""

from __future__ import annotations

from typing import Union

import numpy as np

from ._dtypes import resolve_dtype
from ._metal_backend import MetalBackend
from .ndarray import ndarray

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


def empty(shape, dtype=None) -> ndarray:
    """Return a new array of given shape and type, without initialising entries."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    dtype = resolve_dtype(dtype)
    size = 1
    for s in shape:
        size *= s
    backend = MetalBackend()
    buf = backend.create_buffer(max(size, 1), dtype)
    return ndarray._from_buffer(buf, shape, dtype)


def zeros(shape, dtype=None) -> ndarray:
    """Return a new array of given shape filled with zeros."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    dtype = resolve_dtype(dtype)
    np_data = np.zeros(shape, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(np_data)
    return ndarray._from_buffer(buf, shape, dtype)


def ones(shape, dtype=None) -> ndarray:
    """Return a new array of given shape filled with ones."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    dtype = resolve_dtype(dtype)
    np_data = np.ones(shape, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(np_data)
    return ndarray._from_buffer(buf, shape, dtype)


def full(shape, fill_value, dtype=None) -> ndarray:
    """Return a new array of given shape filled with *fill_value*."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    dtype = resolve_dtype(dtype)
    np_data = np.full(shape, fill_value, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(np_data)
    return ndarray._from_buffer(buf, shape, dtype)


def arange(start_or_stop, stop=None, step=1, dtype=None) -> ndarray:
    """Return evenly spaced values within a given interval."""
    if stop is None:
        start, stop = 0, start_or_stop
    else:
        start = start_or_stop
    np_data = np.arange(start, stop, step)
    if dtype is not None:
        dtype = resolve_dtype(dtype)
        np_data = np_data.astype(dtype)
    else:
        dtype = resolve_dtype(np_data.dtype)
        np_data = np_data.astype(dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(np_data)
    return ndarray._from_buffer(buf, np_data.shape, dtype)


def array(obj, dtype=None) -> ndarray:
    """Create an ndarray from a list, NumPy array, or macmetalpy ndarray."""
    if isinstance(obj, ndarray):
        np_data = obj.get()
    else:
        np_data = np.asarray(obj)

    orig_shape = np_data.shape  # Save BEFORE any shape-mangling
    if dtype is not None:
        dtype = resolve_dtype(dtype)
    else:
        dtype = resolve_dtype(np_data.dtype)
    np_data = np.ascontiguousarray(np_data, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(np_data)
    return ndarray._from_buffer(buf, orig_shape, dtype)


def asarray(obj, dtype=None) -> ndarray:
    """Convert input to an ndarray; no-copy if already the right type/dtype."""
    if isinstance(obj, ndarray):
        if dtype is None or np.dtype(dtype) == obj.dtype:
            return obj
    return array(obj, dtype=dtype)


def zeros_like(a, dtype=None) -> ndarray:
    """Return an array of zeros with the same shape as *a*."""
    if isinstance(a, ndarray):
        shape = a.shape
        dtype = dtype if dtype is not None else a.dtype
    else:
        a = np.asarray(a)
        shape = a.shape
    return zeros(shape, dtype=dtype)


def ones_like(a, dtype=None) -> ndarray:
    """Return an array of ones with the same shape as *a*."""
    if isinstance(a, ndarray):
        shape = a.shape
        dtype = dtype if dtype is not None else a.dtype
    else:
        a = np.asarray(a)
        shape = a.shape
    return ones(shape, dtype=dtype)


def empty_like(a, dtype=None) -> ndarray:
    """Return an uninitialised array with the same shape as *a*."""
    if isinstance(a, ndarray):
        shape = a.shape
        dtype = dtype if dtype is not None else a.dtype
    else:
        a = np.asarray(a)
        shape = a.shape
    return empty(shape, dtype=dtype)


def full_like(a, fill_value, dtype=None) -> ndarray:
    """Return a filled array with the same shape as *a*."""
    if isinstance(a, ndarray):
        shape = a.shape
        dtype = dtype if dtype is not None else a.dtype
    else:
        a = np.asarray(a)
        shape = a.shape
    return full(shape, fill_value, dtype=dtype)


def linspace(start, stop, num=50, dtype=None) -> ndarray:
    """Return evenly spaced numbers over a specified interval."""
    np_data = np.linspace(start, stop, num)
    if dtype is not None:
        dtype = resolve_dtype(dtype)
    else:
        dtype = resolve_dtype(np_data.dtype)
    np_data = np.ascontiguousarray(np_data, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(np_data)
    return ndarray._from_buffer(buf, np_data.shape, dtype)


def eye(N, M=None, dtype=None) -> ndarray:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
    if M is None:
        M = N
    if dtype is not None:
        dtype = resolve_dtype(dtype)
    else:
        dtype = resolve_dtype(None)
    np_data = np.eye(N, M, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(np_data)
    return ndarray._from_buffer(buf, np_data.shape, dtype)


def diag(v, k=0):
    """Extract a diagonal or construct a diagonal array."""
    if isinstance(v, ndarray):
        v_np = v.get()
    else:
        v_np = np.asarray(v)
    result = np.diag(v_np, k=k)
    dtype = resolve_dtype(result.dtype)
    result = np.ascontiguousarray(result, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(result)
    return ndarray._from_buffer(buf, result.shape, dtype)


def identity(n, dtype=None):
    """Return the identity array (alias for eye)."""
    return eye(n, dtype=dtype)


def tri(N, M=None, k=0, dtype=None):
    """Return array with ones at and below the given diagonal."""
    dtype = resolve_dtype(dtype)
    result = np.tri(N, M, k, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(np.ascontiguousarray(result))
    return ndarray._from_buffer(buf, result.shape, dtype)


def triu(m, k=0):
    """Upper triangle of an array."""
    if isinstance(m, ndarray):
        m_np = m.get()
    else:
        m_np = np.asarray(m)
    result = np.triu(m_np, k=k)
    dtype = resolve_dtype(result.dtype)
    result = np.ascontiguousarray(result, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(result)
    return ndarray._from_buffer(buf, result.shape, dtype)


def tril(m, k=0):
    """Lower triangle of an array."""
    if isinstance(m, ndarray):
        m_np = m.get()
    else:
        m_np = np.asarray(m)
    result = np.tril(m_np, k=k)
    dtype = resolve_dtype(result.dtype)
    result = np.ascontiguousarray(result, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(result)
    return ndarray._from_buffer(buf, result.shape, dtype)


def logspace(start, stop, num=50, dtype=None):
    """Return numbers spaced evenly on a log scale."""
    np_data = np.logspace(start, stop, num)
    if dtype is not None:
        dtype = resolve_dtype(dtype)
    else:
        dtype = resolve_dtype(np_data.dtype)
    np_data = np.ascontiguousarray(np_data, dtype=dtype)
    backend = MetalBackend()
    buf = backend.array_to_buffer(np_data)
    return ndarray._from_buffer(buf, np_data.shape, dtype)


def meshgrid(*xi, indexing='xy'):
    """Return coordinate matrices from coordinate vectors."""
    np_arrays = []
    for x in xi:
        if isinstance(x, ndarray):
            np_arrays.append(x.get())
        else:
            np_arrays.append(np.asarray(x))
    results = np.meshgrid(*np_arrays, indexing=indexing)
    return [array(np.ascontiguousarray(r)) for r in results]


def indices(dimensions, dtype=int):
    """Return an array representing the indices of a grid."""
    result = np.indices(dimensions, dtype=dtype)
    return array(result)


def fromfunction(function, shape, dtype=float):
    """Construct array by executing function over each coordinate."""
    result = np.fromfunction(function, shape, dtype=dtype)
    return array(np.ascontiguousarray(result, dtype=resolve_dtype(np.asarray(result).dtype)))


def diagflat(v, k=0):
    """Create a diagonal array with the flattened input as diagonal."""
    if isinstance(v, ndarray):
        v_np = v.get()
    else:
        v_np = np.asarray(v)
    result = np.diagflat(v_np, k=k)
    dtype = resolve_dtype(result.dtype)
    return array(np.ascontiguousarray(result, dtype=dtype))


def vander(x, N=None, increasing=False):
    """Generate a Vandermonde matrix."""
    if isinstance(x, ndarray):
        x_np = x.get()
    else:
        x_np = np.asarray(x)
    result = np.vander(x_np, N=N, increasing=increasing)
    dtype = resolve_dtype(result.dtype)
    return array(np.ascontiguousarray(result, dtype=dtype))


def asanyarray(a, dtype=None):
    """Convert input to an ndarray (pass-through for ndarrays)."""
    return asarray(a, dtype=dtype)
