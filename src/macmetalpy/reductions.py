"""Reduction operations (CuPy-compatible)."""

from __future__ import annotations

import numpy as np


def _copy_to_out(result, out):
    """Copy result data into out array and return out."""
    from . import creation
    # Transfer result to numpy, write into out's buffer
    np_data = result.get()
    out_np = np_data.astype(out.dtype)
    tmp = creation.array(out_np)
    out._buffer = tmp._buffer
    return out


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    """Sum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if dtype is not None:
        a = creation.array(a.get().astype(dtype))
    result = a.sum(axis=axis, keepdims=keepdims)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    """Arithmetic mean of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = a.mean(axis=axis, keepdims=keepdims)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def max(a, axis=None, out=None, keepdims=False):
    """Maximum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = a.max(axis=axis, keepdims=keepdims)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def min(a, axis=None, out=None, keepdims=False):
    """Minimum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = a.min(axis=axis, keepdims=keepdims)
    if out is not None:
        return _copy_to_out(result, out)
    return result


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


def std(a, axis=None, dtype=None, keepdims=False, ddof=0):
    """Standard deviation of array elements.

    Parameters
    ----------
    ddof : int, optional
        Delta degrees of freedom. Default is 0 (population std).
        Use ddof=1 for sample std.
    """
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if ddof == 0:
        return a.std(axis=axis, keepdims=keepdims)
    # For ddof != 0, use CPU fallback for correctness
    result = np.std(a.get(), axis=axis, keepdims=keepdims, ddof=ddof)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def var(a, axis=None, dtype=None, keepdims=False, ddof=0):
    """Variance of array elements.

    Parameters
    ----------
    ddof : int, optional
        Delta degrees of freedom. Default is 0 (population variance).
        Use ddof=1 for sample variance.
    """
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if ddof == 0:
        return a.var(axis=axis, keepdims=keepdims)
    # For ddof != 0, use CPU fallback for correctness
    result = np.var(a.get(), axis=axis, keepdims=keepdims, ddof=ddof)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    """Product of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if dtype is not None:
        a = creation.array(a.get().astype(dtype))
    result = a.prod(axis=axis, keepdims=keepdims)
    if out is not None:
        return _copy_to_out(result, out)
    return result


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
