"""Sorting functions (CuPy-compatible)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray, _c_contiguous_strides, _wrap_np
from . import creation


def _get_np(a):
    """Get numpy data, preferring _np_data for CPU-resident arrays."""
    np_data = a._np_data
    if np_data is not None:
        return np_data
    if a._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
    return a._get_view()


def sort(a, axis=-1, kind=None, order=None, *, stable=None):
    """Return a sorted copy of an array."""
    if stable is not None and kind is None:
        kind = 'stable' if stable else 'quicksort'
    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path for small/medium arrays
    if a.size < 262144:
        return _wrap_np(np.sort(_get_np(a), axis=axis, kind=kind, order=order))

    # GPU path for 1D float/int arrays
    if a.ndim == 1 and a.dtype in (np.float32, np.int32, np.uint32, np.int16, np.uint16):
        return _bitonic_sort_1d(a, return_indices=False)

    # Fall back to CPU for complex cases
    return _wrap_np(np.sort(_get_np(a), axis=axis, kind=kind, order=order))


def argsort(a, axis=-1, kind=None, order=None, *, stable=None):
    """Return the indices that would sort an array."""
    if stable is not None and kind is None:
        kind = 'stable' if stable else 'quicksort'
    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path for small/medium arrays
    if a.size < 262144:
        return _wrap_np(np.argsort(_get_np(a), axis=axis, kind=kind, order=order))

    # GPU path for 1D
    if a.ndim == 1 and a.dtype in (np.float32, np.int32, np.uint32, np.int16, np.uint16):
        return _bitonic_sort_1d(a, return_indices=True)

    return _wrap_np(np.argsort(_get_np(a), axis=axis, kind=kind, order=order))


def _bitonic_sort_1d(a, return_indices=False):
    """GPU bitonic sort for 1-D arrays."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    a_c = a._ensure_contiguous()
    n = a_c.size

    if n <= 1:
        if return_indices:
            return _wrap_np(np.array([0] if n == 1 else [], dtype=np.int32))
        return a.copy()

    # Pad to next power of 2
    n_padded = 1
    while n_padded < n:
        n_padded *= 2

    metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]

    if return_indices:
        # Sort with index tracking
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}

kernel void init_indices(device int *indices [[buffer(0)]],
                          uint id [[thread_position_in_grid]]) {{
    indices[id] = (int)id;
}}

kernel void bitonic_sort_step(device {metal_type} *data [[buffer(0)]],
                               device int *indices [[buffer(1)]],
                               device uint *params [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {{
    uint j = params[0];
    uint k = params[1];
    uint n = params[2];

    uint ixj = id ^ j;
    if (ixj <= id) return;
    if (id >= n || ixj >= n) return;

    bool ascending = ((id & k) == 0);
    {metal_type} a_val = data[id];
    {metal_type} b_val = data[ixj];

    bool should_swap = ascending ? (a_val > b_val) : (a_val < b_val);
    if (should_swap) {{
        data[id] = b_val;
        data[ixj] = a_val;
        int tmp_idx = indices[id];
        indices[id] = indices[ixj];
        indices[ixj] = tmp_idx;
    }}
}}
"""
        backend = MetalBackend()

        # Create padded data buffer
        padded_data = np.empty(n_padded, dtype=a.dtype)
        padded_data[:n] = a_c.get()
        if np.issubdtype(a.dtype, np.floating):
            padded_data[n:] = np.inf
        else:
            padded_data[n:] = np.iinfo(a.dtype).max
        data_buf = backend.array_to_buffer(padded_data)

        # Init indices
        idx_buf = backend.create_buffer(n_padded, np.int32)
        backend.execute_kernel(shader_src, "init_indices", n_padded, [idx_buf])

        # Bitonic sort passes
        k = 2
        while k <= n_padded:
            j = k >> 1
            while j > 0:
                params = np.array([j, k, n_padded], dtype=np.uint32)
                params_buf = backend.array_to_buffer(params)
                backend.execute_kernel(shader_src, "bitonic_sort_step", n_padded, [data_buf, idx_buf, params_buf])
                j >>= 1
            k <<= 1

        # Extract valid indices (view first n elements of the padded buffer, then GPU copy)
        result_idx = ndarray._from_buffer(idx_buf, (n,), np.int32)
        return result_idx.copy()
    else:
        # Sort without indices
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}

kernel void bitonic_sort_step(device {metal_type} *data [[buffer(0)]],
                               device uint *params [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {{
    uint j = params[0];
    uint k = params[1];
    uint n = params[2];

    uint ixj = id ^ j;
    if (ixj <= id) return;
    if (id >= n || ixj >= n) return;

    bool ascending = ((id & k) == 0);
    {metal_type} a_val = data[id];
    {metal_type} b_val = data[ixj];

    bool should_swap = ascending ? (a_val > b_val) : (a_val < b_val);
    if (should_swap) {{
        data[id] = b_val;
        data[ixj] = a_val;
    }}
}}
"""
        backend = MetalBackend()

        padded_data = np.empty(n_padded, dtype=a.dtype)
        padded_data[:n] = a_c.get()
        if np.issubdtype(a.dtype, np.floating):
            padded_data[n:] = np.inf
        else:
            padded_data[n:] = np.iinfo(a.dtype).max
        data_buf = backend.array_to_buffer(padded_data)

        k = 2
        while k <= n_padded:
            j = k >> 1
            while j > 0:
                params = np.array([j, k, n_padded], dtype=np.uint32)
                params_buf = backend.array_to_buffer(params)
                backend.execute_kernel(shader_src, "bitonic_sort_step", n_padded, [data_buf, params_buf])
                j >>= 1
            k <<= 1

        # View first n elements of the padded buffer, then GPU copy to own buffer
        result = ndarray._from_buffer(data_buf, (n,), a.dtype)
        return result.copy()


def unique(ar, return_index=False, return_inverse=False, return_counts=False,
           axis=None, equal_nan=True, *, sorted=True):
    """Find the unique elements of an array."""
    if not isinstance(ar, ndarray):
        ar = creation.asarray(ar)
    np_result = np.unique(
        _get_np(ar),
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        equal_nan=equal_nan,
    )
    if isinstance(np_result, tuple):
        return tuple(_wrap_np(r) for r in np_result)
    return _wrap_np(np_result)


def searchsorted(a, v, side='left', sorter=None):
    """Find indices where elements should be inserted to maintain order."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    v_arr = v if isinstance(v, ndarray) else creation.asarray(np.asarray(v))

    # GPU path for simple case (no sorter, 1D sorted array)
    if sorter is None and a.ndim == 1 and a.dtype in METAL_TYPE_NAMES and v_arr.size > 0:
        a_c = a._ensure_contiguous()
        v_c = v_arr.astype(a.dtype)._ensure_contiguous()

        side_val = 0 if side == 'left' else 1
        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void searchsorted_op(device {metal_type} *sorted_arr [[buffer(0)]],
                              device {metal_type} *values [[buffer(1)]],
                              device int *out [[buffer(2)]],
                              device uint *params [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {{
    uint arr_size = params[0];
    uint side = params[1];
    {metal_type} val = values[id];

    int lo = 0, hi = (int)arr_size;
    while (lo < hi) {{
        int mid = (lo + hi) / 2;
        bool go_right;
        if (side == 0) go_right = sorted_arr[mid] < val;  // left
        else go_right = sorted_arr[mid] <= val;  // right
        if (go_right) lo = mid + 1;
        else hi = mid;
    }}
    out[id] = lo;
}}
"""
        backend = MetalBackend()
        params = np.array([a_c.size, side_val], dtype=np.uint32)
        params_buf = backend.array_to_buffer(params)
        out_buf = backend.create_buffer(v_c.size, np.int32)
        backend.execute_kernel(shader_src, "searchsorted_op", v_c.size, [a_c._buffer, v_c._buffer, out_buf, params_buf])
        result = ndarray._from_buffer(out_buf, v_arr.shape, np.int32)
        return result.astype(np.intp)

    # Fallback
    v_np = v.get() if isinstance(v, ndarray) else np.asarray(v)
    sorter_np = sorter.get() if isinstance(sorter, ndarray) else sorter
    result = np.searchsorted(a.get(), v_np, side=side, sorter=sorter_np)
    return _wrap_np(np.asarray(result))


def lexsort(keys):
    """Perform indirect stable sort using sequence of keys."""
    np_keys = [_get_np(k) if isinstance(k, ndarray) else np.asarray(k) for k in keys]
    result = np.lexsort(np_keys)
    return _wrap_np(result)


def partition(a, kth, axis=-1, kind='introselect', order=None):
    """Return a partitioned copy of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return _wrap_np(np.partition(_get_np(a), kth, axis=axis, kind=kind, order=order))


def argpartition(a, kth, axis=-1, kind='introselect', order=None):
    """Return indices that would partition an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return _wrap_np(np.argpartition(_get_np(a), kth, axis=axis, kind=kind, order=order))


def msort(a):
    """Return a sorted copy along the first axis."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return _wrap_np(np.sort(_get_np(a), axis=0))


def sort_complex(a):
    """Sort a complex array by real then imaginary part.

    Note: Metal does not support complex dtypes, so the result is
    converted to float32 (real part only) when the input is non-complex,
    matching the sorted order that ``np.sort_complex`` would produce.
    """
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_data = a.get()
    result = np.sort_complex(np_data)
    # Metal has no complex dtype support; fall back to float32 for real inputs.
    if not np.issubdtype(result.dtype, np.complexfloating):
        result = result.real.astype(np.float32)
    else:
        result = result.real.astype(np.float32)
    return _wrap_np(result)


# ── NumPy 2 unique variants ──────────────────────────────────────────

from collections import namedtuple

UniqueAllResult = namedtuple("UniqueAllResult", ["values", "indices", "inverse_indices", "counts"])
UniqueCountsResult = namedtuple("UniqueCountsResult", ["values", "counts"])
UniqueInverseResult = namedtuple("UniqueInverseResult", ["values", "inverse_indices"])


def unique_all(a):
    """Return sorted unique elements with indices, inverse indices, and counts."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    vals, indices, inverse, counts = np.unique(
        _get_np(a), return_index=True, return_inverse=True, return_counts=True,
    )
    return UniqueAllResult(
        values=_wrap_np(vals),
        indices=_wrap_np(indices),
        inverse_indices=_wrap_np(inverse),
        counts=_wrap_np(counts),
    )


def unique_counts(a):
    """Return sorted unique elements and their counts."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    vals, counts = np.unique(_get_np(a), return_counts=True)
    return UniqueCountsResult(values=_wrap_np(vals), counts=_wrap_np(counts))


def unique_inverse(a):
    """Return sorted unique elements and indices to reconstruct the input."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    vals, inverse = np.unique(_get_np(a), return_inverse=True)
    return UniqueInverseResult(values=_wrap_np(vals), inverse_indices=_wrap_np(inverse))


def unique_values(a):
    """Return sorted unique elements."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return _wrap_np(np.unique(_get_np(a)))
