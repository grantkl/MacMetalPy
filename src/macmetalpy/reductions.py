"""Reduction operations (CuPy-compatible)."""

from __future__ import annotations

import numpy as np


def sum(a, axis=None, keepdims=False):
    """Sum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.sum(axis=axis, keepdims=keepdims)


def mean(a, axis=None, keepdims=False):
    """Arithmetic mean of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.mean(axis=axis, keepdims=keepdims)


def max(a, axis=None, keepdims=False):
    """Maximum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.max(axis=axis, keepdims=keepdims)


def min(a, axis=None, keepdims=False):
    """Minimum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.min(axis=axis, keepdims=keepdims)


def any(a, axis=None, keepdims=False):
    """Test whether any array element evaluates to True."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.any(axis=axis, keepdims=keepdims)


def all(a, axis=None, keepdims=False):
    """Test whether all array elements evaluate to True."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.all(axis=axis, keepdims=keepdims)


def argmax(a, axis=None):
    """Return indices of the maximum values."""
    from .ndarray import ndarray
    from . import creation
    from ._kernel_cache import KernelCache
    from ._metal_backend import MetalBackend

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if axis is None:
        flat = a.ravel()._ensure_contiguous()
        backend = MetalBackend()
        cache = KernelCache()
        shader = cache.get_shader("axis_reduction", flat._dtype)
        dims_buf = backend.array_to_buffer(np.array([1, flat.size], dtype=np.uint32))
        out_buf = backend.create_buffer(1, np.int32)
        backend.execute_kernel(shader, "argmax_axis", 1, [flat._buffer, out_buf, dims_buf])
        result = ndarray._from_buffer(out_buf, (), np.int32)
        return int(result.get())
    return a._reduce_axis("argmax_axis", axis, out_dtype=np.int32)


def argmin(a, axis=None):
    """Return indices of the minimum values."""
    from .ndarray import ndarray
    from . import creation
    from ._kernel_cache import KernelCache
    from ._metal_backend import MetalBackend

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if axis is None:
        flat = a.ravel()._ensure_contiguous()
        backend = MetalBackend()
        cache = KernelCache()
        shader = cache.get_shader("axis_reduction", flat._dtype)
        dims_buf = backend.array_to_buffer(np.array([1, flat.size], dtype=np.uint32))
        out_buf = backend.create_buffer(1, np.int32)
        backend.execute_kernel(shader, "argmin_axis", 1, [flat._buffer, out_buf, dims_buf])
        result = ndarray._from_buffer(out_buf, (), np.int32)
        return int(result.get())
    return a._reduce_axis("argmin_axis", axis, out_dtype=np.int32)


def std(a, axis=None, keepdims=False):
    """Standard deviation of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.std(axis=axis, keepdims=keepdims)


def var(a, axis=None, keepdims=False):
    """Variance of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.var(axis=axis, keepdims=keepdims)


def prod(a, axis=None, keepdims=False):
    """Product of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.prod(axis=axis, keepdims=keepdims)


def median(a, axis=None):
    """Median of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.median(a.get(), axis=axis)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def percentile(a, q, axis=None):
    """Compute the q-th percentile."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.percentile(a.get(), q, axis=axis)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def cumsum(a, axis=None):
    """Cumulative sum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.cumsum(axis=axis)


def cumprod(a, axis=None):
    """Cumulative product of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.cumprod(axis=axis)


def diff(a, n=1, axis=-1):
    """Calculate n-th discrete difference along the given axis."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.diff(a.get(), n=n, axis=axis)
    return creation.array(result)


def ptp(a, axis=None):
    """Range of values (maximum - minimum) along an axis."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.max(axis=axis)._binary_op(a.min(axis=axis), "sub_op")


def quantile(a, q, axis=None):
    """Compute the q-th quantile."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.quantile(a.get(), q, axis=axis)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def average(a, axis=None, weights=None):
    """Compute weighted average."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    w = weights.get() if isinstance(weights, ndarray) else weights
    result = np.average(a.get(), axis=axis, weights=w)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


# ------------------------------------------------------------------
# Internal helpers called by ndarray reduction methods
# ------------------------------------------------------------------

THREADGROUP_SIZE = 256


def _reduce_gpu(arr, op_name):
    """Full-array GPU reduction. Returns scalar ndarray."""
    from ._kernel_cache import KernelCache
    from ._metal_backend import MetalBackend
    from . import creation

    backend = MetalBackend()
    cache = KernelCache()

    a = arr._ensure_contiguous()
    size = a.size

    num_groups = (size + THREADGROUP_SIZE - 1) // THREADGROUP_SIZE

    shader_src = cache.get_shader("reduction", a.dtype)

    # Buffer containing the element count (uint32)
    n_buf = backend.array_to_buffer(np.array([size], dtype=np.uint32))

    # Output buffer for partial results (one per threadgroup)
    out_buf = backend.create_buffer(num_groups, a.dtype)

    backend.execute_kernel(shader_src, op_name, size, [a._buffer, out_buf, n_buf])

    # Final reduction on CPU
    partials = out_buf.contents[:num_groups].copy()

    if op_name == "reduce_sum":
        result_val = partials.sum()
    elif op_name == "reduce_max":
        result_val = partials.max()
    elif op_name == "reduce_min":
        result_val = partials.min()
    else:
        raise ValueError(f"Unknown reduction op: {op_name!r}")

    return creation.array(np.array(result_val, dtype=a.dtype))


def _reduce_axis_cpu(arr, np_func, axis, keepdims):
    """Axis-specific reduction via CPU fallback."""
    from . import creation

    np_result = np_func(arr.get(), axis=axis, keepdims=keepdims)
    return creation.array(np_result)
