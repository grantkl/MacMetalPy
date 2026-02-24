"""Reduction operations (CuPy-compatible)."""

from __future__ import annotations

import numpy as np


def _copy_to_out(result, out):
    """Copy result data into out array and return out (GPU-resident)."""
    from .ndarray import ndarray
    from . import creation
    if not isinstance(result, ndarray):
        result = creation.asarray(result)
    # Cast on GPU if needed and write buffer into out
    casted = result.astype(out.dtype)._ensure_contiguous()
    out._adopt_buffer(casted._buffer)
    return out


def _cpu_view(a):
    """Get a zero-copy numpy view of a's data (syncs if GPU-resident)."""
    if a._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
    return a._get_view()


def _apply_where(a, where, fill_value):
    """Apply a where mask, replacing masked-out elements with fill_value on GPU."""
    from .ndarray import ndarray
    from . import creation
    from ._kernel_cache import KernelCache
    from ._metal_backend import MetalBackend

    if not isinstance(where, ndarray):
        where = creation.asarray(where)

    # Create fill array
    fill_arr = creation.full(a.shape, fill_value, dtype=a.dtype)

    # Use GPU where kernel: result = where ? a : fill_value
    cond_int = where.astype(np.int32)._ensure_contiguous()
    a_c = a._ensure_contiguous()
    fill_c = fill_arr._ensure_contiguous()

    backend = MetalBackend()
    cache = KernelCache()
    shader = cache.get_shader("where", a.dtype)
    out_buf = backend.create_buffer(a.size, a.dtype)
    backend.execute_kernel(shader, "where_op", a.size, [cond_int._buffer, a_c._buffer, fill_c._buffer, out_buf])
    return ndarray._from_buffer(out_buf, a.shape, a.dtype)


def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    """Sum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path for small/medium arrays without complex params
    if a.size < _GPU_THRESHOLD_MEMORY and where is np._NoValue:
        np_data = a._np_data
        if np_data is None:
            from ._metal_backend import MetalBackend
            MetalBackend().synchronize()
            np_data = a._get_view()
        r = np.sum(np_data, axis=axis, dtype=dtype, keepdims=keepdims)
        if initial is not np._NoValue:
            r = r + initial
        if not isinstance(r, np.ndarray):
            r = np.asarray(r)
        result = ndarray._from_np_direct(r)
        if out is not None:
            return _copy_to_out(result, out)
        return result
    if where is not np._NoValue:
        a = _apply_where(a, where, 0)
    if dtype is not None:
        a = a.astype(dtype)
    result = a.sum(axis=axis, keepdims=keepdims)
    if initial is not np._NoValue:
        result = result + initial
    if out is not None:
        return _copy_to_out(result, out)
    return result


def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=np._NoValue):
    """Arithmetic mean of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path for small/medium arrays
    if a.size < _GPU_THRESHOLD_MEMORY:
        np_data = a._np_data
        if np_data is None:
            from ._metal_backend import MetalBackend
            MetalBackend().synchronize()
            np_data = a._get_view()
        if where is np._NoValue and dtype is None:
            result_np = np.mean(np_data, axis=axis, keepdims=keepdims)
        else:
            kwargs = dict(axis=axis, keepdims=keepdims)
            if dtype is not None:
                kwargs['dtype'] = dtype
            if where is not np._NoValue:
                w = where.get() if isinstance(where, ndarray) else np.asarray(where)
                kwargs['where'] = w
            result_np = np.mean(np_data, **kwargs)
        if not isinstance(result_np, np.ndarray):
            result_np = np.asarray(result_np)
        result = ndarray._from_np_direct(result_np)
        if out is not None:
            return _copy_to_out(result, out)
        return result
    if where is not np._NoValue:
        w = where.get() if isinstance(where, ndarray) else np.asarray(where)
        result_np = np.mean(a.get(), axis=axis, keepdims=keepdims, where=w)
        if not isinstance(result_np, np.ndarray):
            result_np = np.array(result_np)
        result = creation.array(result_np)
    else:
        result = a.mean(axis=axis, keepdims=keepdims)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def max(a, axis=None, out=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    """Maximum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path — use _np_data directly to skip _cpu_view overhead
    if a.size < _GPU_THRESHOLD_MEMORY and where is np._NoValue:
        np_data = a._np_data
        if np_data is None:
            from ._metal_backend import MetalBackend
            MetalBackend().synchronize()
            np_data = a._get_view()
        if initial is np._NoValue:
            r = np.max(np_data, axis=axis, keepdims=keepdims)
        else:
            r = np.maximum(np.max(np_data, axis=axis, keepdims=keepdims), initial)
        if not isinstance(r, np.ndarray):
            r = np.asarray(r)
        result = ndarray._from_np_direct(r)
        if out is not None:
            return _copy_to_out(result, out)
        return result
    if where is not np._NoValue:
        a = _apply_where(a, where, -np.inf)
    result = a.max(axis=axis, keepdims=keepdims)
    if initial is not np._NoValue:
        result = result._binary_op(creation.full(result.shape, initial, dtype=result.dtype), "max_op")
    if out is not None:
        return _copy_to_out(result, out)
    return result


def min(a, axis=None, out=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    """Minimum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path — use _np_data directly to skip _cpu_view overhead
    if a.size < _GPU_THRESHOLD_MEMORY and where is np._NoValue:
        np_data = a._np_data
        if np_data is None:
            from ._metal_backend import MetalBackend
            MetalBackend().synchronize()
            np_data = a._get_view()
        if initial is np._NoValue:
            r = np.min(np_data, axis=axis, keepdims=keepdims)
        else:
            r = np.minimum(np.min(np_data, axis=axis, keepdims=keepdims), initial)
        if not isinstance(r, np.ndarray):
            r = np.asarray(r)
        result = ndarray._from_np_direct(r)
        if out is not None:
            return _copy_to_out(result, out)
        return result
    if where is not np._NoValue:
        a = _apply_where(a, where, np.inf)
    result = a.min(axis=axis, keepdims=keepdims)
    if initial is not np._NoValue:
        result = result._binary_op(creation.full(result.shape, initial, dtype=result.dtype), "min_op")
    if out is not None:
        return _copy_to_out(result, out)
    return result


def any(a, axis=None, out=None, keepdims=False, where=np._NoValue):
    """Test whether any array element evaluates to True."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path
    if a.size < _GPU_THRESHOLD_MEMORY:
        np_data = a._np_data if a._np_data is not None else _cpu_view(a)
        kwargs = dict(axis=axis, keepdims=keepdims)
        if where is not np._NoValue:
            kwargs['where'] = where.get() if isinstance(where, ndarray) else np.asarray(where)
        result_np = np.any(np_data, **kwargs)
        if not isinstance(result_np, np.ndarray):
            result_np = np.array(result_np)
        result = ndarray._from_np_direct(result_np)
        if out is not None:
            return _copy_to_out(result, out)
        return result
    if where is not np._NoValue:
        w = where.get() if isinstance(where, ndarray) else np.asarray(where)
        result_np = np.any(a.get(), axis=axis, keepdims=keepdims, where=w)
        if not isinstance(result_np, np.ndarray):
            result_np = np.array(result_np)
        result = ndarray._from_np_direct(result_np)
    else:
        result = a.any(axis=axis, keepdims=keepdims)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def all(a, axis=None, out=None, keepdims=False, where=np._NoValue):
    """Test whether all array elements evaluate to True."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path
    if a.size < _GPU_THRESHOLD_MEMORY:
        np_data = a._np_data if a._np_data is not None else _cpu_view(a)
        kwargs = dict(axis=axis, keepdims=keepdims)
        if where is not np._NoValue:
            kwargs['where'] = where.get() if isinstance(where, ndarray) else np.asarray(where)
        result_np = np.all(np_data, **kwargs)
        if not isinstance(result_np, np.ndarray):
            result_np = np.array(result_np)
        result = ndarray._from_np_direct(result_np)
        if out is not None:
            return _copy_to_out(result, out)
        return result
    if where is not np._NoValue:
        w = where.get() if isinstance(where, ndarray) else np.asarray(where)
        result_np = np.all(a.get(), axis=axis, keepdims=keepdims, where=w)
        if not isinstance(result_np, np.ndarray):
            result_np = np.array(result_np)
        result = ndarray._from_np_direct(result_np)
    else:
        result = a.all(axis=axis, keepdims=keepdims)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def _parallel_argmax(flat):
    """Two-pass parallel argmax on a flattened 1-D GPU array."""
    from .ndarray import ndarray
    from ._kernel_cache import KernelCache
    from ._metal_backend import MetalBackend

    backend = MetalBackend()
    cache = KernelCache()
    n = flat.size
    block_size = 1024
    n_blocks = (n + block_size - 1) // block_size
    shader = cache.get_shader("parallel_reduction", flat._dtype)

    # Pass 1: each thread finds argmax in its block
    params_buf = backend.array_to_buffer(np.array([n, block_size], dtype=np.uint32))
    partial_vals = backend.create_buffer(n_blocks, flat._dtype)
    partial_idx = backend.create_buffer(n_blocks, np.int32)
    backend.execute_kernel(
        shader, "par_argmax_block", n_blocks,
        [flat._buffer, partial_vals, partial_idx, params_buf],
    )

    # Pass 2: reduce partial results
    out_buf = backend.create_buffer(1, np.int32)
    final_params = backend.array_to_buffer(np.array([n_blocks, 0], dtype=np.uint32))
    backend.execute_kernel(
        shader, "par_argmax_final", 1,
        [partial_vals, partial_idx, out_buf, final_params],
    )
    return ndarray._from_buffer(out_buf, (), np.int32)


def _parallel_argmin(flat):
    """Two-pass parallel argmin on a flattened 1-D GPU array."""
    from .ndarray import ndarray
    from ._kernel_cache import KernelCache
    from ._metal_backend import MetalBackend

    backend = MetalBackend()
    cache = KernelCache()
    n = flat.size
    block_size = 1024
    n_blocks = (n + block_size - 1) // block_size
    shader = cache.get_shader("parallel_reduction", flat._dtype)

    # Pass 1: each thread finds argmin in its block
    params_buf = backend.array_to_buffer(np.array([n, block_size], dtype=np.uint32))
    partial_vals = backend.create_buffer(n_blocks, flat._dtype)
    partial_idx = backend.create_buffer(n_blocks, np.int32)
    backend.execute_kernel(
        shader, "par_argmin_block", n_blocks,
        [flat._buffer, partial_vals, partial_idx, params_buf],
    )

    # Pass 2: reduce partial results
    out_buf = backend.create_buffer(1, np.int32)
    final_params = backend.array_to_buffer(np.array([n_blocks, 0], dtype=np.uint32))
    backend.execute_kernel(
        shader, "par_argmin_final", 1,
        [partial_vals, partial_idx, out_buf, final_params],
    )
    return ndarray._from_buffer(out_buf, (), np.int32)


_GPU_THRESHOLD_MEMORY = 4194304  # 4M — reductions are memory-bound, CPU wins


def argmax(a, axis=None, out=None, keepdims=False):
    """Return indices of the maximum values."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if axis is None:
        # CPU fallback for small arrays
        if a.size < _GPU_THRESHOLD_MEMORY:
            np_data = a._np_data if a._np_data is not None else _cpu_view(a)
            result = int(np.argmax(np_data))
            if keepdims:
                r = creation.array(np.array(result).reshape((1,) * a.ndim))
                return _copy_to_out(r, out) if out is not None else r
            if out is not None:
                return _copy_to_out(creation.array(np.intp(result)), out)
            return result
        flat = a.ravel()._ensure_contiguous()
        raw = _parallel_argmax(flat)
        if keepdims:
            result = raw.astype(np.intp).reshape((1,) * a.ndim)
        else:
            if out is not None:
                result = raw.astype(np.intp)
                return _copy_to_out(result, out)
            return int(raw.get())
        if out is not None:
            return _copy_to_out(result, out)
        return result
    result = a._reduce_axis("argmax_axis", axis, out_dtype=np.int32)
    if keepdims:
        shape = list(a.shape)
        shape[axis % a.ndim] = 1
        result = result.reshape(tuple(shape))
    if out is not None:
        return _copy_to_out(result, out)
    return result


def argmin(a, axis=None, out=None, keepdims=False):
    """Return indices of the minimum values."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if axis is None:
        # CPU fallback for small arrays
        if a.size < _GPU_THRESHOLD_MEMORY:
            np_data = a._np_data if a._np_data is not None else _cpu_view(a)
            result = int(np.argmin(np_data))
            if keepdims:
                r = creation.array(np.array(result).reshape((1,) * a.ndim))
                return _copy_to_out(r, out) if out is not None else r
            if out is not None:
                return _copy_to_out(creation.array(np.intp(result)), out)
            return result
        flat = a.ravel()._ensure_contiguous()
        raw = _parallel_argmin(flat)
        if keepdims:
            result = raw.astype(np.intp).reshape((1,) * a.ndim)
        else:
            if out is not None:
                result = raw.astype(np.intp)
                return _copy_to_out(result, out)
            return int(raw.get())
        if out is not None:
            return _copy_to_out(result, out)
        return result
    result = a._reduce_axis("argmin_axis", axis, out_dtype=np.int32)
    if keepdims:
        shape = list(a.shape)
        shape[axis % a.ndim] = 1
        result = result.reshape(tuple(shape))
    if out is not None:
        return _copy_to_out(result, out)
    return result


def std(a, axis=None, dtype=None, out=None, keepdims=False, ddof=0, where=np._NoValue, *, correction=None, mean=None):
    """Standard deviation of array elements.

    Parameters
    ----------
    ddof : int, optional
        Delta degrees of freedom. Default is 0 (population std).
        Use ddof=1 for sample std.
    correction : int, optional
        NumPy 2.0 alias for ddof. If provided, overrides ddof.
    mean : array_like, optional
        Accepted for API compatibility but currently ignored.
    """
    from .ndarray import ndarray
    from . import creation

    if correction is not None:
        ddof = correction
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path for small/medium arrays
    if a.size < _GPU_THRESHOLD_MEMORY:
        np_data = a._np_data
        if np_data is None:
            from ._metal_backend import MetalBackend
            MetalBackend().synchronize()
            np_data = a._get_view()
        kwargs = dict(axis=axis, keepdims=keepdims, ddof=ddof)
        if where is not np._NoValue:
            kwargs['where'] = where.get() if isinstance(where, ndarray) else np.asarray(where)
        result = np.std(np_data, **kwargs)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        r = ndarray._from_np_direct(result)
        return _copy_to_out(r, out) if out is not None else r
    if where is not np._NoValue:
        w = where.get() if isinstance(where, ndarray) else np.asarray(where)
        result = np.std(a.get(), axis=axis, keepdims=keepdims, ddof=ddof, where=w)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        r = creation.array(result)
        return _copy_to_out(r, out) if out is not None else r
    if ddof == 0:
        r = a.std(axis=axis, keepdims=keepdims)
        return _copy_to_out(r, out) if out is not None else r

    # GPU two-pass for ddof != 0
    m = a.mean(axis=axis, keepdims=True)  # GPU mean
    diff = a - m  # GPU subtract (broadcast)
    sq = diff * diff  # GPU square
    variance = sq.sum(axis=axis, keepdims=keepdims)

    # Compute count
    if axis is None:
        n = a.size
    else:
        n = a.shape[axis if axis >= 0 else axis + a.ndim]

    variance_result = variance / (n - ddof)
    from . import math_ops
    r = math_ops.sqrt(variance_result)
    return _copy_to_out(r, out) if out is not None else r


def var(a, axis=None, dtype=None, out=None, keepdims=False, ddof=0, where=np._NoValue, *, correction=None, mean=None):
    """Variance of array elements.

    Parameters
    ----------
    ddof : int, optional
        Delta degrees of freedom. Default is 0 (population variance).
        Use ddof=1 for sample variance.
    correction : int, optional
        NumPy 2.0 alias for ddof. If provided, overrides ddof.
    mean : array_like, optional
        Accepted for API compatibility but currently ignored.
    """
    from .ndarray import ndarray
    from . import creation

    if correction is not None:
        ddof = correction
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path for small/medium arrays
    if a.size < _GPU_THRESHOLD_MEMORY:
        np_data = a._np_data
        if np_data is None:
            from ._metal_backend import MetalBackend
            MetalBackend().synchronize()
            np_data = a._get_view()
        kwargs = dict(axis=axis, keepdims=keepdims, ddof=ddof)
        if where is not np._NoValue:
            kwargs['where'] = where.get() if isinstance(where, ndarray) else np.asarray(where)
        result = np.var(np_data, **kwargs)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        r = ndarray._from_np_direct(result)
        return _copy_to_out(r, out) if out is not None else r
    if where is not np._NoValue:
        w = where.get() if isinstance(where, ndarray) else np.asarray(where)
        result = np.var(a.get(), axis=axis, keepdims=keepdims, ddof=ddof, where=w)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        r = creation.array(result)
        return _copy_to_out(r, out) if out is not None else r
    if ddof == 0:
        r = a.var(axis=axis, keepdims=keepdims)
        return _copy_to_out(r, out) if out is not None else r

    # GPU two-pass for ddof != 0
    m = a.mean(axis=axis, keepdims=True)  # GPU mean
    diff = a - m  # GPU subtract (broadcast)
    sq = diff * diff  # GPU square
    variance = sq.sum(axis=axis, keepdims=keepdims)

    # Compute count
    if axis is None:
        n = a.size
    else:
        n = a.shape[axis if axis >= 0 else axis + a.ndim]

    r = variance / (n - ddof)
    return _copy_to_out(r, out) if out is not None else r


def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    """Product of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path
    if a.size < _GPU_THRESHOLD_MEMORY and where is np._NoValue:
        np_data = a._np_data
        if np_data is None:
            from ._metal_backend import MetalBackend
            MetalBackend().synchronize()
            np_data = a._get_view()
        r = np.prod(np_data, axis=axis, dtype=dtype, keepdims=keepdims)
        if initial is not np._NoValue:
            r = r * initial
        result = ndarray._from_np_direct(np.asarray(r))
        if out is not None:
            return _copy_to_out(result, out)
        return result
    if where is not np._NoValue:
        a = _apply_where(a, where, 1)
    if dtype is not None:
        a = a.astype(dtype)
    result = a.prod(axis=axis, keepdims=keepdims)
    if initial is not np._NoValue:
        result = result * initial
    if out is not None:
        return _copy_to_out(result, out)
    return result


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Median of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_data = a._np_data if a._np_data is not None else _cpu_view(a)
    result = np.median(np_data, axis=axis, keepdims=keepdims)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    result = ndarray._from_np_direct(result)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False):
    """Compute the q-th percentile."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_data = a._np_data if a._np_data is not None else _cpu_view(a)
    result = np.percentile(np_data, q, axis=axis, keepdims=keepdims)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    result = ndarray._from_np_direct(result)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def cumsum(a, axis=None, dtype=None, out=None):
    """Cumulative sum of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path
    if a.size < _GPU_THRESHOLD_MEMORY:
        np_data = a._np_data if a._np_data is not None else _cpu_view(a)
        r = np.cumsum(np_data, axis=axis, dtype=dtype)
        result = ndarray._from_np_direct(r)
        if out is not None:
            return _copy_to_out(result, out)
        return result
    if dtype is not None:
        a = a.astype(dtype)
    result = a.cumsum(axis=axis)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def cumprod(a, axis=None, dtype=None, out=None):
    """Cumulative product of array elements."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path
    if a.size < _GPU_THRESHOLD_MEMORY:
        np_data = a._np_data if a._np_data is not None else _cpu_view(a)
        r = np.cumprod(np_data, axis=axis, dtype=dtype)
        result = ndarray._from_np_direct(r)
        if out is not None:
            return _copy_to_out(result, out)
        return result
    if dtype is not None:
        a = a.astype(dtype)
    result = a.cumprod(axis=axis)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    """Calculate n-th discrete difference along the given axis."""
    from .ndarray import ndarray
    from . import creation
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path for small/medium arrays
    if a.size < _GPU_THRESHOLD_MEMORY:
        np_data = a._np_data if a._np_data is not None else _cpu_view(a)
        if prepend is np._NoValue and append is np._NoValue:
            return ndarray._from_np_direct(np.diff(np_data, n=n, axis=axis))
        kwargs = dict(n=n, axis=axis)
        if prepend is not np._NoValue:
            kwargs['prepend'] = prepend.get() if isinstance(prepend, ndarray) else prepend
        if append is not np._NoValue:
            kwargs['append'] = append.get() if isinstance(append, ndarray) else append
        return ndarray._from_np_direct(np.diff(np_data, **kwargs))

    # Handle prepend/append
    if prepend is not np._NoValue or append is not np._NoValue:
        # Fall back to CPU for prepend/append
        kwargs = dict(n=n, axis=axis)
        if prepend is not np._NoValue:
            kwargs['prepend'] = prepend.get() if isinstance(prepend, ndarray) else prepend
        if append is not np._NoValue:
            kwargs['append'] = append.get() if isinstance(append, ndarray) else append
        result = np.diff(a.get(), **kwargs)
        return creation.array(result)

    # GPU path for simple diff (n=1, no prepend/append)
    if n == 1 and a.ndim == 1:
        a_c = a._ensure_contiguous()
        if a_c.size <= 1:
            return creation.array(np.array([], dtype=a.dtype))

        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void diff_1d(device {metal_type} *a [[buffer(0)]],
                    device {metal_type} *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {{
    out[id] = a[id + 1] - a[id];
}}
"""
        backend = MetalBackend()
        out_size = a_c.size - 1
        out_buf = backend.create_buffer(out_size, a.dtype)
        backend.execute_kernel(shader_src, "diff_1d", out_size, [a_c._buffer, out_buf])
        result = ndarray._from_buffer(out_buf, (out_size,), a.dtype)
    else:
        # For n>1 or multi-dimensional, fall back to CPU
        result = creation.array(np.diff(a.get(), n=n, axis=axis))

    return result


def divmod(x1, x2):
    """Return element-wise quotient and remainder simultaneously."""
    from .ndarray import ndarray
    from . import creation
    from . import math_ops

    if not isinstance(x1, ndarray):
        x1 = creation.asarray(x1)
    if not isinstance(x2, ndarray):
        x2 = creation.asarray(x2)
    q = x1._binary_op(x2, "floor_divide_op")
    r = math_ops.mod(x1, x2)
    return q, r


def ptp(a, axis=None, out=None, keepdims=False):
    """Range of values (maximum - minimum) along an axis."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path — single numpy call instead of two separate reductions
    if a._np_data is not None or a.size < 4194304:
        np_data = a._np_data
        if np_data is None:
            from ._metal_backend import MetalBackend
            MetalBackend().synchronize()
            np_data = a._get_view()
        result = ndarray._from_np_direct(np.ptp(np_data, axis=axis, keepdims=keepdims))
    else:
        result = a.max(axis=axis, keepdims=keepdims) - a.min(axis=axis, keepdims=keepdims)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False):
    """Compute the q-th quantile."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_data = a._np_data if a._np_data is not None else _cpu_view(a)
    result = np.quantile(np_data, q, axis=axis, keepdims=keepdims)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    result = ndarray._from_np_direct(result)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def average(a, axis=None, weights=None, returned=False, keepdims=False):
    """Compute weighted average."""
    from .ndarray import ndarray
    from . import creation

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_data = a._np_data
    if np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
        np_data = a._get_view()
    w = weights.get() if isinstance(weights, ndarray) else weights
    result = np.average(np_data, axis=axis, weights=w, returned=returned, keepdims=keepdims)
    if returned:
        avg, sum_w = result
        if not isinstance(avg, np.ndarray):
            avg = np.array(avg)
        if not isinstance(sum_w, np.ndarray):
            sum_w = np.array(sum_w)
        return ndarray._from_np_direct(avg), ndarray._from_np_direct(sum_w)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return ndarray._from_np_direct(result)


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
    from .ndarray import ndarray

    np_result = np_func(_cpu_view(arr), axis=axis, keepdims=keepdims)
    return ndarray._from_np_direct(np.asarray(np_result))


# ── NumPy 2 cumulative aliases ───────────────────────────────────────

def cumulative_sum(a, axis=None, dtype=None):
    """Cumulative sum (NumPy 2 alias for cumsum)."""
    return cumsum(a, axis=axis, dtype=dtype)


def cumulative_prod(a, axis=None, dtype=None):
    """Cumulative product (NumPy 2 alias for cumprod)."""
    return cumprod(a, axis=axis, dtype=dtype)
