"""Indexing routines (CuPy-compatible)."""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def _get_np(a):
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
    return _get_np(a)


# ------------------------------------------------------------------ Take / Put


def take(a, indices, axis=None, out=None, mode='raise'):
    """Take elements from an array along an axis."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    ind = indices.get() if isinstance(indices, ndarray) else np.asarray(indices)

    # CPU fast path for small/medium arrays
    if a.size < 4194304:
        if a._np_data is None:
            from ._metal_backend import MetalBackend as _MB
            _MB().synchronize()
        result = ndarray._from_np_direct(np.take(a._get_view(), ind, axis=axis, mode=mode))
        if out is not None:
            out._np_data = result.astype(out.dtype)._np_data
            out._buffer = None
            return out
        return result

    # GPU path for flat take (axis=None) or 1D
    if (axis is None or a.ndim == 1) and a.dtype in METAL_TYPE_NAMES and ind.size > 0:
        a_flat = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
        ind_flat = ind.ravel().astype(np.int32)

        # Handle negative indices and modes
        if mode == 'wrap':
            ind_flat = ind_flat % a_flat.size
        elif mode == 'clip':
            ind_flat = np.clip(ind_flat, 0, a_flat.size - 1)
        else:
            # mode='raise': convert negative indices, then check bounds
            neg_mask = ind_flat < 0
            if neg_mask.any():
                ind_flat = ind_flat.copy()
                ind_flat[neg_mask] += a_flat.size

        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void gather_op(device {metal_type} *src [[buffer(0)]],
                       device int *indices [[buffer(1)]],
                       device {metal_type} *out [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {{
    out[id] = src[indices[id]];
}}
"""
        backend = MetalBackend()
        ind_buf = backend.array_to_buffer(ind_flat)
        out_buf = backend.create_buffer(ind_flat.size, a.dtype)
        backend.execute_kernel(shader_src, "gather_op", ind_flat.size, [a_flat._buffer, ind_buf, out_buf])
        result = ndarray._from_buffer(out_buf, ind.shape, a.dtype)
        if out is not None:
            out._adopt_buffer(result.astype(out.dtype)._ensure_contiguous()._buffer)
            return out
        return result

    # Fallback
    ind_np = ind
    result = np.take(a.get(), ind_np, axis=axis, mode=mode)
    result = creation.array(result)
    if out is not None:
        out._adopt_buffer(result.astype(out.dtype)._ensure_contiguous()._buffer)
        return out
    return result


def take_along_axis(arr, indices, axis):
    """Take values from the input array by matching 1d index and data slices along the given axis."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    ind_np = _get_np(indices) if isinstance(indices, ndarray) else indices
    result = np.take_along_axis(_get_np(arr), ind_np, axis)
    return ndarray._from_np_direct(result)


def put(a, ind, v, mode='raise'):
    """Replaces specified elements of an array with given values. Mutates *a* in place."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    ind_np = _get_np(ind) if isinstance(ind, ndarray) else np.asarray(ind, dtype=np.int32)
    v_np = _get_np(v) if isinstance(v, ndarray) else np.asarray(v)

    # CPU fast path for small/medium arrays
    if a.size < 4194304:
        if a._np_data is None:
            from ._metal_backend import MetalBackend as _MB
            _MB().synchronize()
            a._np_data = a._get_view().copy()
            a._buffer = None
        np.put(a._np_data.ravel(), ind_np, v_np, mode=mode)
        return None

    if a.dtype in METAL_TYPE_NAMES and ind_np.size > 0:
        ind_flat = ind_np.ravel().astype(np.int32)
        v_flat = np.broadcast_to(v_np, ind_flat.shape).astype(a.dtype).ravel()

        if mode == 'wrap':
            ind_flat = ind_flat % a.size
        elif mode == 'clip':
            ind_flat = np.clip(ind_flat, 0, a.size - 1)

        # GPU scatter has race conditions with duplicate indices;
        # fall back to CPU if duplicates exist.
        if len(np.unique(ind_flat)) < len(ind_flat):
            tmp = a.get()
            np.put(tmp, ind_flat, v_flat, mode='raise')
            a.set(tmp)
            return None

        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void scatter_op(device {metal_type} *dst [[buffer(0)]],
                        device int *indices [[buffer(1)]],
                        device {metal_type} *values [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {{
    dst[indices[id]] = values[id];
}}
"""
        a_c = a._ensure_contiguous()
        backend = MetalBackend()
        ind_buf = backend.array_to_buffer(ind_flat)
        val_buf = backend.array_to_buffer(v_flat)
        backend.execute_kernel(shader_src, "scatter_op", ind_flat.size, [a_c._buffer, ind_buf, val_buf])
        # Update a's buffer
        a._adopt_buffer(a_c._buffer)
        return None

    # Fallback
    tmp = a.get()
    np.put(tmp, ind_np, v_np, mode=mode)
    a.set(tmp)
    return None


def put_along_axis(arr, indices, values, axis):
    """Put values into the destination array by matching 1d index and data slices along the given axis."""
    indices_np = _get_np(indices) if isinstance(indices, ndarray) else indices
    values_np = _get_np(values) if isinstance(values, ndarray) else values
    if arr._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
        arr._np_data = arr._get_view().copy()
        arr._buffer = None
    np.put_along_axis(arr._np_data, indices_np, values_np, axis)
    return None


def putmask(a, mask, values):
    """Changes elements of an array based on conditional and input values. Mutates *a*."""
    mask_np = _get_np(mask) if isinstance(mask, ndarray) else mask
    val_np = _get_np(values) if isinstance(values, ndarray) else values
    if a._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
        a._np_data = a._get_view().copy()
        a._buffer = None
    np.putmask(a._np_data, mask_np, val_np)
    return None


def place(arr, mask, vals):
    """Change elements of an array based on conditional and input values. Mutates *arr*."""
    mask_np = _get_np(mask) if isinstance(mask, ndarray) else mask
    val_np = _get_np(vals) if isinstance(vals, ndarray) else vals
    if arr._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
        arr._np_data = arr._get_view().copy()
        arr._buffer = None
    np.place(arr._np_data, mask_np, val_np)
    return None


# ------------------------------------------------------------------ Selection


def choose(a, choices, out=None, mode='raise'):
    """Construct an array from an index array and a set of arrays to choose from."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_choices = [_get_np(c) if isinstance(c, ndarray) else c for c in choices]
    result = ndarray._from_np_direct(np.choose(_get_np(a), np_choices, mode=mode))
    if out is not None:
        out._np_data = result.astype(out.dtype)._np_data
        out._buffer = None
        return out
    return result


def compress(condition, a, axis=None, out=None):
    """Return selected slices of an array along given axis."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    cond_np = _get_np(condition) if isinstance(condition, ndarray) else condition
    result = ndarray._from_np_direct(np.compress(cond_np, _get_np(a), axis=axis))
    if out is not None:
        out._np_data = result.astype(out.dtype)._np_data
        out._buffer = None
        return out
    return result


def select(condlist, choicelist, default=0):
    """Return an array drawn from elements in choicelist, depending on conditions."""
    np_conds = [_get_np(c) if isinstance(c, ndarray) else c for c in condlist]
    np_choices = [_get_np(ch) if isinstance(ch, ndarray) else ch for ch in choicelist]
    result = np.select(np_conds, np_choices, default=default)
    return ndarray._from_np_direct(result)


def extract(condition, arr):
    """Return the elements of an array that satisfy some condition."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    cond_np = _get_np(condition) if isinstance(condition, ndarray) else condition
    result = np.extract(cond_np, _get_np(arr))
    return ndarray._from_np_direct(result)


# ------------------------------------------------------------------ Index arrays


def diag_indices(n, ndim=2):
    """Return the indices to access the main diagonal of an array."""
    return tuple(ndarray._from_np_direct(x) for x in np.diag_indices(n, ndim))


def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an n-dimensional array."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    return tuple(ndarray._from_np_direct(x) for x in np.diag_indices_from(_get_np(arr)))


def tril_indices(n, k=0, m=None):
    """Return the indices for the lower-triangle of an (n, m) array."""
    return tuple(ndarray._from_np_direct(x) for x in np.tril_indices(n, k, m))


def tril_indices_from(arr, k=0):
    """Return the indices for the lower-triangle of arr."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    return tuple(ndarray._from_np_direct(x) for x in np.tril_indices_from(_get_np(arr), k))


def triu_indices(n, k=0, m=None):
    """Return the indices for the upper-triangle of an (n, m) array."""
    return tuple(ndarray._from_np_direct(x) for x in np.triu_indices(n, k, m))


def triu_indices_from(arr, k=0):
    """Return the indices for the upper-triangle of arr."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    return tuple(ndarray._from_np_direct(x) for x in np.triu_indices_from(_get_np(arr), k))


# ------------------------------------------------------------------ Conversion


def ravel_multi_index(multi_index, dims):
    """Converts a tuple of index arrays into an array of flat indices."""
    np_mi = [_get_np(x) if isinstance(x, ndarray) else x for x in multi_index]
    result = np.ravel_multi_index(np_mi, dims)
    return ndarray._from_np_direct(np.asarray(result))


def unravel_index(indices, shape):
    """Converts a flat index or array of flat indices into a tuple of coordinate arrays."""
    ind_np = _get_np(indices) if isinstance(indices, ndarray) else indices
    return tuple(ndarray._from_np_direct(np.asarray(x)) for x in np.unravel_index(ind_np, shape))


# ------------------------------------------------------------------ Fill


def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given array. Mutates *a* in place."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    # Fall back to CPU when val is an array (not a scalar), since the GPU
    # shader only reads val[0] and can't cycle through multiple values.
    val_is_array = isinstance(val, (list, tuple, np.ndarray, ndarray))
    if val_is_array:
        val_cpu = val.get() if isinstance(val, ndarray) else val
        tmp = a.get()
        np.fill_diagonal(tmp, val_cpu, wrap=wrap)
        a.set(tmp)
        return None

    # CPU fast path for small/medium arrays
    if a.size < 4194304:
        if a._np_data is None:
            from ._metal_backend import MetalBackend as _MB
            _MB().synchronize()
            a._np_data = a._get_view().copy()
            a._buffer = None
        np.fill_diagonal(a._np_data, val, wrap=wrap)
        return None

    if a.ndim == 2 and a.dtype in METAL_TYPE_NAMES and not wrap:
        rows, cols = a.shape
        diag_len = min(rows, cols)
        a_c = a._ensure_contiguous()

        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void fill_diag_op(device {metal_type} *a [[buffer(0)]],
                          device {metal_type} *val [[buffer(1)]],
                          device uint *dims [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {{
    uint cols = dims[0];
    a[id * cols + id] = val[0];
}}
"""
        backend = MetalBackend()
        val_arr = np.array([val], dtype=a.dtype)
        val_buf = backend.array_to_buffer(val_arr)
        dims = np.array([cols], dtype=np.uint32)
        dims_buf = backend.array_to_buffer(dims)
        backend.execute_kernel(shader_src, "fill_diag_op", diag_len, [a_c._buffer, val_buf, dims_buf])
        a._adopt_buffer(a_c._buffer)
        return None

    tmp = a.get()
    np.fill_diagonal(tmp, val, wrap=wrap)
    a.set(tmp)
    return None


# ------------------------------------------------------------------ Search


def nonzero(a):
    """Return the indices of the elements that are non-zero."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return tuple(ndarray._from_np_direct(x) for x in np.nonzero(_get_np(a)))


def flatnonzero(a):
    """Return indices that are non-zero in the flattened version of a."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return ndarray._from_np_direct(np.flatnonzero(_get_np(a)).astype(np.intp))


def argwhere(a):
    """Find the indices of array elements that are non-zero, grouped by element."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return ndarray._from_np_direct(np.argwhere(_get_np(a)))


# ------------------------------------------------------------------ Advanced


def mask_indices(n, mask_func, k=0):
    """Return the indices to access (n, n) arrays, given a masking function."""
    def _np_mask_func(m, k=0):
        r = mask_func(m, k=k)
        return _get_np(r) if isinstance(r, ndarray) else r
    result = np.mask_indices(n, _np_mask_func, k=k)
    return tuple(ndarray._from_np_direct(np.asarray(x)) for x in result)


def ix_(*args):
    """Construct an open mesh from multiple sequences."""
    np_args = [_get_np(a) if isinstance(a, ndarray) else a for a in args]
    return tuple(ndarray._from_np_direct(x) for x in np.ix_(*np_args))
