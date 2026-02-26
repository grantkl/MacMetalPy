"""Array manipulation functions (CuPy-compatible)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


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


def tile(a, reps):
    """Construct an array by repeating a the number of times given by reps."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path — tile is memory-bound, CPU is faster for reasonable sizes
    if a._np_data is not None or a.size < 4194304:
        np_data = _get_np(a)
        return ndarray._from_np_direct(np.tile(np_data, reps))

    # GPU path for 1D tile
    if a.ndim == 1 and isinstance(reps, (int, np.integer)) and a.dtype in METAL_TYPE_NAMES:
        a_c = a._ensure_contiguous()
        src_size = a_c.size
        out_size = src_size * int(reps)

        if out_size == 0:
            return ndarray._from_np_direct(np.tile(_get_np(a), reps))

        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void tile_1d(device {metal_type} *src [[buffer(0)]],
                     device {metal_type} *dst [[buffer(1)]],
                     device uint *params [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {{
    uint src_size = params[0];
    dst[id] = src[id % src_size];
}}
"""
        backend = MetalBackend()
        params = np.array([src_size], dtype=np.uint32)
        params_buf = backend.array_to_buffer(params)
        out_buf = backend.create_buffer(out_size, a.dtype)
        backend.execute_kernel(shader_src, "tile_1d", out_size, [a_c._buffer, out_buf, params_buf])
        return ndarray._from_buffer(out_buf, (out_size,), a.dtype)

    return ndarray._from_np_direct(np.tile(_get_np(a), reps))


def repeat(a, repeats, axis=None):
    """Repeat elements of an array."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path
    if a._np_data is not None or a.size < 4194304:
        np_data = _get_np(a)
        return ndarray._from_np_direct(np.repeat(np_data, repeats, axis=axis))

    # GPU path for 1D uniform repeat
    if (axis is None or a.ndim == 1) and isinstance(repeats, (int, np.integer)) and a.dtype in METAL_TYPE_NAMES:
        flat = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
        rep = int(repeats)
        out_size = flat.size * rep

        if out_size == 0:
            return ndarray._from_np_direct(np.repeat(_get_np(a), repeats, axis=axis))

        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void repeat_1d(device {metal_type} *src [[buffer(0)]],
                       device {metal_type} *dst [[buffer(1)]],
                       device uint *params [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {{
    uint rep = params[0];
    dst[id] = src[id / rep];
}}
"""
        backend = MetalBackend()
        params = np.array([rep], dtype=np.uint32)
        params_buf = backend.array_to_buffer(params)
        out_buf = backend.create_buffer(out_size, a.dtype)
        backend.execute_kernel(shader_src, "repeat_1d", out_size, [flat._buffer, out_buf, params_buf])
        return ndarray._from_buffer(out_buf, (out_size,), a.dtype)

    return ndarray._from_np_direct(np.repeat(_get_np(a), repeats, axis=axis))


def flip(a, axis=None):
    """Reverse the order of elements along the given axis."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path — flip is a pure memory op, CPU is always faster
    if a._np_data is not None or a.size < 4194304:
        return ndarray._from_np_direct(np.flip(_get_np(a), axis=axis))

    # GPU path for 1D
    if a.ndim == 1 or axis is None:
        flat = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
        n = flat.size
        if n == 0:
            return ndarray._from_np_direct(np.flip(_get_np(a), axis=axis))

        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void flip_1d(device {metal_type} *src [[buffer(0)]],
                     device {metal_type} *dst [[buffer(1)]],
                     device uint *params [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {{
    uint n = params[0];
    dst[id] = src[n - 1 - id];
}}
"""
        backend = MetalBackend()
        params = np.array([n], dtype=np.uint32)
        params_buf = backend.array_to_buffer(params)
        out_buf = backend.create_buffer(n, a.dtype)
        backend.execute_kernel(shader_src, "flip_1d", n, [flat._buffer, out_buf, params_buf])
        return ndarray._from_buffer(out_buf, a.shape, a.dtype)

    # For multi-dimensional flip, fall back
    return ndarray._from_np_direct(np.flip(_get_np(a), axis=axis))


def roll(a, shift, axis=None):
    """Roll array elements along a given axis."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path
    if a._np_data is not None or a.size < 4194304:
        np_data = _get_np(a)
        return ndarray._from_np_direct(np.roll(np_data, shift, axis=axis))

    # GPU path for 1D or flat roll
    if (axis is None or a.ndim == 1) and a.dtype in METAL_TYPE_NAMES:
        flat = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
        n = flat.size
        if n == 0:
            return flat.copy().reshape(a.shape)

        # Normalize shift
        s = int(shift) if not isinstance(shift, (list, tuple)) else int(shift[0]) if axis is None else int(shift)
        s = s % n
        if s == 0:
            from . import math_ops
            return math_ops.copy(a) if axis is not None else flat.copy().reshape(a.shape)

        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void roll_1d(device {metal_type} *src [[buffer(0)]],
                     device {metal_type} *dst [[buffer(1)]],
                     device uint *params [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {{
    uint n = params[0];
    uint shift = params[1];
    uint src_idx = (id + n - shift) % n;
    dst[id] = src[src_idx];
}}
"""
        backend = MetalBackend()
        params = np.array([n, s], dtype=np.uint32)
        params_buf = backend.array_to_buffer(params)
        out_buf = backend.create_buffer(n, a.dtype)
        backend.execute_kernel(shader_src, "roll_1d", n, [flat._buffer, out_buf, params_buf])
        return ndarray._from_buffer(out_buf, a.shape, a.dtype)

    return ndarray._from_np_direct(np.roll(_get_np(a), shift, axis=axis))


def _wrap_split_results(results):
    """Wrap numpy split results into ndarrays with minimal overhead."""
    out = []
    for r in results:
        arr = ndarray.__new__(ndarray)
        arr._buffer = None
        arr._np_data = r
        arr._shape = r.shape
        arr._dtype = r.dtype
        if r.flags['C_CONTIGUOUS']:
            # Inline C-contiguous strides computation
            shape = r.shape
            if not shape:
                arr._strides = ()
            else:
                strides = [1] * len(shape)
                for i in range(len(shape) - 2, -1, -1):
                    strides[i] = strides[i + 1] * shape[i + 1]
                arr._strides = tuple(strides)
        else:
            itemsize = r.itemsize
            arr._strides = tuple(s // itemsize for s in r.strides)
        arr._offset = 0
        arr._base = None
        out.append(arr)
    return out


def split(a, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_data = _get_np(a)
    return _wrap_split_results(np.split(np_data, indices_or_sections, axis=axis))


def array_split(a, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays (allows unequal division)."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_data = _get_np(a)
    return _wrap_split_results(np.array_split(np_data, indices_or_sections, axis=axis))


def squeeze(a, axis=None):
    """Remove single-dimensional entries from the shape of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.squeeze(axis=axis)


def ravel(a, order='C'):
    """Return a contiguous flattened array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.ravel()


def moveaxis(a, source, destination):
    """Move axes of an array to new positions."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return ndarray._from_np_direct(np.moveaxis(_get_np(a), source, destination))


def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path
    if a._np_data is not None or a.size < 4194304:
        return ndarray._from_np_direct(np.swapaxes(_get_np(a), axis1, axis2))
    # GPU path: transpose with swapped axes, then ensure contiguous
    n = a.ndim
    axis1 = axis1 % n if axis1 < 0 else axis1
    axis2 = axis2 % n if axis2 < 0 else axis2
    axes = list(range(n))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return a.transpose(axes)._ensure_contiguous()


def broadcast_to(array, shape):
    """Broadcast an array to a new shape."""
    a = array
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path — broadcast_to returns a read-only view, store as-is
    out_size = 1
    for s in shape:
        out_size *= s
    if a._np_data is not None or out_size < 4194304:
        return ndarray._from_np_direct(np.broadcast_to(_get_np(a), shape))

    # For scalar or simple 1D broadcast, use GPU
    if a.ndim <= 1 and a.dtype in METAL_TYPE_NAMES and a.size > 0:
        # Validate broadcast compatibility: last dim of shape must match src size (or src is scalar)
        src_size = a.size
        if src_size != 1 and (len(shape) == 0 or shape[-1] != src_size):
            # Fall through to numpy which will raise the proper ValueError
            return ndarray._from_np_direct(np.broadcast_to(_get_np(a), shape))

        a_c = a._ensure_contiguous()
        out_size = 1
        for s in shape:
            out_size *= s

        if out_size == 0:
            return ndarray._from_np_direct(np.broadcast_to(_get_np(a), shape))
        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void broadcast_op(device {metal_type} *src [[buffer(0)]],
                          device {metal_type} *dst [[buffer(1)]],
                          device uint *params [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {{
    uint src_size = params[0];
    dst[id] = src[id % src_size];
}}
"""
        backend = MetalBackend()
        params = np.array([src_size], dtype=np.uint32)
        params_buf = backend.array_to_buffer(params)
        out_buf = backend.create_buffer(out_size, a.dtype)
        backend.execute_kernel(shader_src, "broadcast_op", out_size, [a_c._buffer, out_buf, params_buf])
        return ndarray._from_buffer(out_buf, tuple(shape), a.dtype)

    return ndarray._gpu_broadcast_to(a, shape)


# ------------------------------------------------------------------ extended manipulation

def reshape(a, newshape, order='C', *, copy=None):
    """Give a new shape to an array without changing its data."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.reshape(newshape)


def transpose(a, axes=None):
    """Permute the dimensions of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.transpose(axes)


def rollaxis(a, axis, start=0):
    """Roll the specified axis backwards."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return ndarray._from_np_direct(np.rollaxis(_get_np(a), axis, start))


def atleast_1d(*arys):
    """Convert inputs to arrays with at least one dimension."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = creation.asarray(a)
        if a.ndim == 0:
            results.append(a.reshape(1))
        else:
            results.append(a)
    return results[0] if len(results) == 1 else results


def atleast_2d(*arys):
    """Convert inputs to arrays with at least two dimensions."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = creation.asarray(a)
        if a.ndim == 0:
            results.append(a.reshape(1, 1))
        elif a.ndim == 1:
            results.append(a.reshape(1, a.shape[0]))
        else:
            results.append(a)
    return results[0] if len(results) == 1 else results


def atleast_3d(*arys):
    """Convert inputs to arrays with at least three dimensions."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = creation.asarray(a)
        if a.ndim == 0:
            results.append(a.reshape(1, 1, 1))
        elif a.ndim == 1:
            results.append(a.reshape(1, a.shape[0], 1))
        elif a.ndim == 2:
            results.append(a.reshape(a.shape[0], a.shape[1], 1))
        else:
            results.append(a)
    return results[0] if len(results) == 1 else results


def dstack(tup):
    """Stack arrays in sequence depth wise (along third axis)."""
    np_arrays = [_get_np(a) if isinstance(a, ndarray) else np.asarray(a) for a in tup]
    return ndarray._from_np_direct(np.dstack(np_arrays))


def column_stack(tup):
    """Stack 1-D arrays as columns into a 2-D array."""
    np_arrays = [_get_np(a) if isinstance(a, ndarray) else np.asarray(a) for a in tup]
    return ndarray._from_np_direct(np.column_stack(np_arrays))


def concat(arrays, axis=0, *, dtype=None, casting='same_kind'):
    """Alias for concatenate."""
    np_arrays = [_get_np(a) if isinstance(a, ndarray) else np.asarray(a) for a in arrays]
    result = ndarray._from_np_direct(np.concatenate(np_arrays, axis=axis))
    if dtype is not None:
        result = result.astype(dtype)
    return result


def dsplit(ary, indices_or_sections):
    """Split array along third axis."""
    if not isinstance(ary, ndarray):
        ary = creation.asarray(ary)
    np_data = _get_np(ary)
    return _wrap_split_results(np.dsplit(np_data, indices_or_sections))


def hsplit(ary, indices_or_sections):
    """Split array along second axis."""
    if not isinstance(ary, ndarray):
        ary = creation.asarray(ary)
    np_data = _get_np(ary)
    return _wrap_split_results(np.hsplit(np_data, indices_or_sections))


def vsplit(ary, indices_or_sections):
    """Split array along first axis."""
    if not isinstance(ary, ndarray):
        ary = creation.asarray(ary)
    np_data = _get_np(ary)
    return _wrap_split_results(np.vsplit(np_data, indices_or_sections))


def delete(arr, obj, axis=None):
    """Return a new array with sub-arrays along an axis deleted."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    return ndarray._from_np_direct(np.delete(_get_np(arr), obj, axis=axis))


def append(arr, values, axis=None):
    """Append values to the end of an array."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    val_np = _get_np(values) if isinstance(values, ndarray) else np.asarray(values)
    return ndarray._from_np_direct(np.append(_get_np(arr), val_np, axis=axis))


def resize(a, new_shape):
    """Return a new array with the given shape."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return ndarray._from_np_direct(np.resize(_get_np(a), new_shape))


def trim_zeros(filt, trim='fb'):
    """Trim the leading and/or trailing zeros from a 1-D array."""
    if not isinstance(filt, ndarray):
        filt = creation.asarray(filt)
    return ndarray._from_np_direct(np.asarray(np.trim_zeros(_get_np(filt), trim)))


def fliplr(m):
    """Reverse the order of elements along axis 1 (left/right)."""
    if not isinstance(m, ndarray):
        m = creation.asarray(m)
    return ndarray._from_np_direct(np.fliplr(_get_np(m)))


def flipud(m):
    """Reverse the order of elements along axis 0 (up/down)."""
    if not isinstance(m, ndarray):
        m = creation.asarray(m)
    return ndarray._from_np_direct(np.flipud(_get_np(m)))


def rot90(m, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the plane specified by axes."""
    if not isinstance(m, ndarray):
        m = creation.asarray(m)
    # np.rot90 returns a view — store as non-contiguous, lazy contiguity
    return ndarray._from_np_direct(np.rot90(_get_np(m), k=k, axes=axes))


def broadcast_arrays(*args):
    """Broadcast any number of arrays against each other."""
    arrays = [a if isinstance(a, ndarray) else creation.asarray(a) for a in args]
    # Use numpy broadcast_arrays — returns read-only views (no copy)
    np_arrays = [_get_np(a) for a in arrays]
    np_results = np.broadcast_arrays(*np_arrays)
    return [ndarray._from_np_direct(r) for r in np_results]


def copyto(dst, src):
    """Copy values from one array to another."""
    if not isinstance(src, ndarray):
        src = creation.asarray(src)
    # CPU fast path
    if src._np_data is not None or src.size < 4194304:
        result = ndarray._from_np_direct(_get_np(src).astype(dst.dtype, copy=False))
        dst._np_data = result._np_data
        dst._buffer = None
        dst._strides = result._strides
        dst._offset = 0
        return
    # GPU path: convert src to dst dtype and copy buffer contents
    converted = src.astype(dst.dtype)._ensure_contiguous()
    dst._adopt_buffer(converted._buffer)
    dst._offset = 0
    dst._strides = converted._strides


def pad(array, pad_width, mode='constant', **kwargs):
    """Pad an array."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(array, ndarray):
        array = creation.asarray(array)

    # CPU fast path — pad is memory-bound, CPU handles it well
    if array._np_data is not None or array.size < 4194304:
        return ndarray._from_np_direct(np.pad(_get_np(array), pad_width, mode=mode, **kwargs))

    # GPU path for 1D constant padding
    if array.ndim == 1 and mode == 'constant' and array.dtype in METAL_TYPE_NAMES:
        constant_values = kwargs.get('constant_values', 0)
        if isinstance(constant_values, (int, float)):
            pad_val = constant_values
        elif isinstance(constant_values, (list, tuple)):
            if isinstance(constant_values[0], (list, tuple)):
                pad_val = constant_values[0][0]  # simplified
            else:
                pad_val = constant_values[0]
        else:
            pad_val = 0

        pw = np.asarray(pad_width)
        if pw.ndim == 1:
            left, right = int(pw[0]), int(pw[1])
        elif pw.ndim == 0:
            left = right = int(pw)
        else:
            left, right = int(pw[0, 0]), int(pw[0, 1])

        a_c = array._ensure_contiguous()
        out_size = a_c.size + left + right

        metal_type = METAL_TYPE_NAMES[np.dtype(array.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void pad_1d(device {metal_type} *src [[buffer(0)]],
                    device {metal_type} *dst [[buffer(1)]],
                    device uint *params [[buffer(2)]],
                    device {metal_type} *fill [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {{
    uint left = params[0];
    uint src_size = params[1];
    if (id < left) dst[id] = fill[0];
    else if (id < left + src_size) dst[id] = src[id - left];
    else dst[id] = fill[0];
}}
"""
        backend = MetalBackend()
        params = np.array([left, a_c.size], dtype=np.uint32)
        params_buf = backend.array_to_buffer(params)
        fill_arr = np.array([pad_val], dtype=array.dtype)
        fill_buf = backend.array_to_buffer(fill_arr)
        out_buf = backend.create_buffer(out_size, array.dtype)
        backend.execute_kernel(shader_src, "pad_1d", out_size, [a_c._buffer, out_buf, params_buf, fill_buf])
        return ndarray._from_buffer(out_buf, (out_size,), array.dtype)

    return ndarray._from_np_direct(np.pad(_get_np(array), pad_width, mode=mode, **kwargs))


def block(arrays):
    """Assemble an ndarray from nested lists of blocks."""
    def _to_numpy(obj):
        if isinstance(obj, ndarray):
            return _get_np(obj)
        if isinstance(obj, list):
            return [_to_numpy(item) for item in obj]
        return np.asarray(obj)

    np_arrays = _to_numpy(arrays)
    return ndarray._from_np_direct(np.block(np_arrays))


def insert(arr, obj, values, axis=None):
    """Insert values along the given axis before the given indices."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    val_np = _get_np(values) if isinstance(values, ndarray) else np.asarray(values)
    return ndarray._from_np_direct(np.insert(_get_np(arr), obj, val_np, axis=axis))


def broadcast_shapes(*args):
    """Broadcast shapes and return the resulting shape."""
    return np.broadcast_shapes(*args)


def asfortranarray(a, dtype=None):
    """Return an array laid out in Fortran order in memory.

    Since Metal is row-major only, this returns a contiguous (C-order) copy
    with the correct values and dtype, matching the NumPy interface.
    """
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_data = _get_np(a)
    if dtype is None and np_data.flags['C_CONTIGUOUS']:
        return a
    target = np_data if dtype is None else np.ascontiguousarray(np_data, dtype=dtype)
    if target is np_data:
        return a
    return ndarray._from_np_direct(target)


# ── NumPy 2 array manipulation functions ─────────────────────────────

def astype(a, dtype):
    """Function form of ndarray.astype — cast array to a specified type."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.astype(dtype)


def matrix_transpose(a):
    """Transpose the last two dimensions of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return swapaxes(a, -2, -1)


def permute_dims(a, axes):
    """Permute the dimensions of an array (alias for transpose)."""
    return transpose(a, axes)


def unstack(a, axis=0):
    """Split an array along an axis into a list of arrays."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    n = a.shape[axis]
    data = _get_np(a)
    results = []
    for i in range(n):
        slices = [slice(None)] * a.ndim
        slices[axis] = i
        results.append(ndarray._from_np_direct(data[tuple(slices)]))
    return results
