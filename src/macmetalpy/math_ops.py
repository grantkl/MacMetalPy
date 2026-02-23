"""Module-level math functions (CuPy-compatible)."""

from __future__ import annotations

import numpy as np

from .ndarray import (ndarray, _fast_unary, _fast_binary,
    _UOP_SQRT, _UOP_EXP, _UOP_LOG, _UOP_ABS, _UOP_SIGN, _UOP_FLOOR,
    _UOP_CEIL, _UOP_SIN, _UOP_COS, _UOP_TAN, _UOP_ASIN, _UOP_ACOS,
    _UOP_ATAN, _UOP_SINH, _UOP_COSH, _UOP_TANH, _UOP_LOG2, _UOP_LOG10,
    _UOP_SQUARE, _UOP_NEGATIVE, _OP_POW)
from . import creation


def sqrt(x, **kwargs):
    """Element-wise square root."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_SQRT)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("sqrt_op")


def exp(x, **kwargs):
    """Element-wise exponential."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_EXP)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("exp_op")


def log(x, **kwargs):
    """Element-wise natural logarithm."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_LOG)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("log_op")


def abs(x, **kwargs):
    """Element-wise absolute value."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_ABS)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("abs_op")


def power(x1, x2, **kwargs):
    """Element-wise power."""
    if _fast_binary is not None and type(x1) is ndarray:
        r = _fast_binary(x1, x2, _OP_POW)
        if r is not None:
            return r
    if type(x1) is not ndarray:
        x1 = creation.asarray(x1)
    return x1._binary_op(x2, "pow_op")


def dot(a, b, out=None):
    """Dot product of two arrays."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    # CPU fast path for small/medium arrays
    if a.size < 4194304 and b.size < 4194304:
        def _v(x):
            if x._np_data is None:
                from ._metal_backend import MetalBackend
                MetalBackend().synchronize()
            return x._get_view()
        r = np.dot(_v(a), _v(b))
        if isinstance(r, np.ndarray):
            result = ndarray._from_np_direct(r)
        else:
            result = ndarray._from_np_direct(np.asarray(r))
        if out is not None:
            out._np_data = result.astype(out.dtype).get()
            out._buffer = None
            return out
        return result
    # 1-D dot product: use fused multiply-reduce kernel
    if a.ndim == 1 and b.ndim == 1:
        result = a._reduce_dot(b)
    else:
        # 2-D: use matmul
        result = a.__matmul__(b)
    if out is not None:
        out._np_data = result.astype(out.dtype).get()
        out._buffer = None
        return out
    return result


def where(condition, x=None, y=None):
    """Return elements chosen from x or y depending on condition (GPU-native)."""
    if x is None and y is None:
        raise NotImplementedError("where() with only condition is not supported")
    from ._broadcasting import broadcast_shapes, needs_broadcast
    from ._dtypes import result_dtype
    from ._kernel_cache import KernelCache
    from ._metal_backend import MetalBackend

    if not isinstance(condition, ndarray):
        condition = creation.asarray(condition)
    if type(x) is not ndarray:
        x = creation.asarray(x if not isinstance(x, (int, float)) else
                              np.array(x, dtype=np.float32))
    if not isinstance(y, ndarray):
        y = creation.asarray(y if not isinstance(y, (int, float)) else
                              np.array(y, dtype=np.float32))

    # CPU fast path for small/medium arrays
    max_size = max(condition.size, x.size, y.size)
    if max_size < 4194304:
        def _v(a):
            if a._np_data is None:
                MetalBackend().synchronize()
            return a._get_view()
        return ndarray._from_np_direct(np.where(_v(condition), _v(x), _v(y)))

    # Determine output dtype from x and y
    rdtype = result_dtype(x.dtype, y.dtype)
    x = x.astype(rdtype) if x.dtype != rdtype else x
    y = y.astype(rdtype) if y.dtype != rdtype else y

    # Broadcast all three to the same shape
    out_shape = broadcast_shapes(broadcast_shapes(condition.shape, x.shape), y.shape)
    if condition.shape != out_shape:
        condition = ndarray._gpu_broadcast_to(condition, out_shape)
    if x.shape != out_shape:
        x = ndarray._gpu_broadcast_to(x, out_shape)
    if y.shape != out_shape:
        y = ndarray._gpu_broadcast_to(y, out_shape)

    # Convert condition to int32 for the kernel (nonzero = true)
    cond_int = condition.astype(np.int32)._ensure_contiguous()
    x = x._ensure_contiguous()
    y = y._ensure_contiguous()

    backend = MetalBackend()
    cache = KernelCache()
    shader = cache.get_shader("where", rdtype)
    out_buf = backend.create_buffer(x.size, rdtype)
    backend.execute_kernel(shader, "where_op", x.size,
                           [cond_int._buffer, x._buffer, y._buffer, out_buf])
    return ndarray._from_buffer(out_buf, out_shape, rdtype)


def clip(a, a_min=None, a_max=None, out=None, *, min=None, max=None):
    """Clip (limit) the values in an array (GPU-native)."""
    from ._kernel_cache import KernelCache
    from ._metal_backend import MetalBackend

    if min is not None and a_min is None:
        a_min = min
    if max is not None and a_max is None:
        a_max = max
    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path for small/medium arrays
    if a.size < 4194304:
        np_data = a._np_data
        if np_data is None:
            MetalBackend().synchronize()
            np_data = a._get_view()
        lo_val = a_min._np_data if isinstance(a_min, ndarray) and a_min._np_data is not None else (a_min.get() if isinstance(a_min, ndarray) else a_min)
        hi_val = a_max._np_data if isinstance(a_max, ndarray) and a_max._np_data is not None else (a_max.get() if isinstance(a_max, ndarray) else a_max)
        result = ndarray._from_np_direct(np.clip(np_data, lo_val, hi_val))
        if out is not None:
            out._np_data = result._np_data
            out._buffer = None
            return out
        return result

    lo = creation.full(a.shape, a_min, dtype=a.dtype) if isinstance(a_min, (int, float)) else creation.asarray(a_min).astype(a.dtype)
    hi = creation.full(a.shape, a_max, dtype=a.dtype) if isinstance(a_max, (int, float)) else creation.asarray(a_max).astype(a.dtype)

    a = a._ensure_contiguous()
    lo = lo._ensure_contiguous()
    hi = hi._ensure_contiguous()

    backend = MetalBackend()
    cache = KernelCache()
    shader = cache.get_shader("clip", a.dtype)
    out_buf = backend.create_buffer(a.size, a.dtype)
    backend.execute_kernel(shader, "clip_op", a.size,
                           [a._buffer, lo._buffer, hi._buffer, out_buf])
    result = ndarray._from_buffer(out_buf, a.shape, a.dtype)
    if out is not None:
        out._adopt_buffer(result._ensure_contiguous()._buffer)
        return out
    return result


def concatenate(arrays, axis=0, out=None, *, dtype=None, casting='same_kind'):
    """Join a sequence of arrays along an existing axis."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    arrs = []
    for arr in arrays:
        if isinstance(arr, ndarray):
            arrs.append(arr)
        else:
            arrs.append(creation.asarray(arr))

    # CPU fast path for small arrays
    total_size = sum(a.size for a in arrs) if arrs else 0
    if total_size < 262144 or all(a._np_data is not None for a in arrs):
        np_arrays = [a._np_data if a._np_data is not None else a.get() for a in arrs]
        result = ndarray._from_np_direct(np.concatenate(np_arrays, axis=axis))
        if dtype is not None:
            result = result.astype(dtype)
        if out is not None:
            out._np_data = result.astype(out.dtype).get()
            out._buffer = None
            return out
        return result

    # For 1D concatenation, use GPU copy with offsets
    if arrs and all(a.ndim == 1 for a in arrs) and axis == 0:
        total_size = sum(a.size for a in arrs)
        out_dtype = dtype if dtype is not None else arrs[0].dtype

        metal_type = METAL_TYPE_NAMES[np.dtype(out_dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void copy_offset(device {metal_type} *src [[buffer(0)]],
                         device {metal_type} *dst [[buffer(1)]],
                         device uint *params [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {{
    uint offset = params[0];
    uint size = params[1];
    if (id < size) dst[offset + id] = src[id];
}}
"""
        backend = MetalBackend()
        out_buf = backend.create_buffer(total_size, out_dtype)
        offset = 0
        for arr in arrs:
            arr_c = arr.astype(out_dtype)._ensure_contiguous()
            params = np.array([offset, arr_c.size], dtype=np.uint32)
            params_buf = backend.array_to_buffer(params)
            backend.execute_kernel(shader_src, "copy_offset", arr_c.size, [arr_c._buffer, out_buf, params_buf])
            offset += arr_c.size
        result = ndarray._from_buffer(out_buf, (total_size,), out_dtype)
        if out is not None:
            out._adopt_buffer(result._ensure_contiguous()._buffer)
            return out
        return result

    # Fallback for multi-dimensional
    np_arrays = [a.get() for a in arrs]
    result_np = np.concatenate(np_arrays, axis=axis)
    result = ndarray._from_np_direct(result_np)
    if dtype is not None:
        result = result.astype(dtype)
    if out is not None:
        out._np_data = result.astype(out.dtype).get()
        out._buffer = None
        return out
    return result


def stack(arrays, axis=0, out=None, dtype=None, casting='same_kind'):
    """Join a sequence of arrays along a new axis."""
    np_arrays = []
    for arr in arrays:
        if isinstance(arr, ndarray):
            np_arrays.append(arr._np_data if arr._np_data is not None else arr.get())
        else:
            np_arrays.append(np.asarray(arr))
    result_np = np.stack(np_arrays, axis=axis)
    result = ndarray._from_np_direct(result_np)
    if dtype is not None:
        result = result.astype(dtype)
    if out is not None:
        out._np_data = result.get()
        out._buffer = None
        return out
    return result


def vstack(arrays, dtype=None, casting='same_kind'):
    """Stack arrays vertically (row-wise)."""
    np_arrays = []
    for arr in arrays:
        if isinstance(arr, ndarray):
            np_arrays.append(arr._np_data if arr._np_data is not None else arr.get())
        else:
            np_arrays.append(np.asarray(arr))
    result_np = np.vstack(np_arrays)
    result = ndarray._from_np_direct(result_np)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def hstack(arrays, dtype=None, casting='same_kind'):
    """Stack arrays horizontally (column-wise)."""
    np_arrays = []
    for arr in arrays:
        if isinstance(arr, ndarray):
            np_arrays.append(arr._np_data if arr._np_data is not None else arr.get())
        else:
            np_arrays.append(np.asarray(arr))
    result_np = np.hstack(np_arrays)
    result = ndarray._from_np_direct(result_np)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def sign(x, **kwargs):
    """Element-wise sign function."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_SIGN)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("sign_op")


def floor(x, **kwargs):
    """Element-wise floor."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_FLOOR)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("floor_op")


def ceil(x, **kwargs):
    """Element-wise ceiling."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_CEIL)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("ceil_op")


def sin(x, **kwargs):
    """Element-wise sine."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_SIN)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("sin_op")


def cos(x, **kwargs):
    """Element-wise cosine."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_COS)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("cos_op")


def tan(x, **kwargs):
    """Element-wise tangent."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_TAN)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("tan_op")


def arcsin(x, **kwargs):
    """Element-wise inverse sine."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_ASIN)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("asin_op")


def arccos(x, **kwargs):
    """Element-wise inverse cosine."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_ACOS)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("acos_op")


def arctan(x, **kwargs):
    """Element-wise inverse tangent."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_ATAN)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("atan_op")


def sinh(x, **kwargs):
    """Element-wise hyperbolic sine."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_SINH)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("sinh_op")


def cosh(x, **kwargs):
    """Element-wise hyperbolic cosine."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_COSH)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("cosh_op")


def tanh(x, **kwargs):
    """Element-wise hyperbolic tangent."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_TANH)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("tanh_op")


def log2(x, **kwargs):
    """Element-wise base-2 logarithm."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_LOG2)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("log2_op")


def log10(x, **kwargs):
    """Element-wise base-10 logarithm."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_LOG10)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("log10_op")


def square(x, **kwargs):
    """Element-wise square."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_SQUARE)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("square_op")


def negative(x, **kwargs):
    """Element-wise negation."""
    if _fast_unary is not None and type(x) is ndarray:
        r = _fast_unary(x, _UOP_NEGATIVE)
        if r is not None:
            return r
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_op("negative_op")


def around(x, decimals=0, out=None):
    """Round to the given number of decimals."""
    if type(x) is not ndarray:
        x = creation.asarray(x)
    # CPU fast path for small/medium arrays
    if x._np_data is not None and x.size < 4194304:
        result = ndarray._from_np_direct(np.around(x._np_data, decimals=decimals))
    elif decimals == 0:
        result = x._unary_op("rint_op")
    else:
        # Scale, round, unscale: round(x * 10^d) / 10^d
        factor = creation.asarray(np.array(10.0 ** decimals, dtype=x.dtype))
        scaled = x * factor
        rounded = scaled._unary_op("rint_op")
        result = rounded / factor
    if out is not None:
        out._adopt_buffer(result._ensure_contiguous()._buffer)
        return out
    return result


def round_(x, decimals=0, out=None):
    """Round to the given number of decimals (alias for around)."""
    return around(x, decimals=decimals, out=out)


round = around


def mod(x1, x2, **kwargs):
    """Element-wise remainder of division (NumPy-compatible: sign of divisor)."""
    if type(x1) is not ndarray:
        x1 = creation.asarray(x1)
    if type(x2) is not ndarray:
        x2 = creation.asarray(x2)
    # NumPy mod: a - floor(a/b)*b  (result has sign of divisor)
    # C fmod gives sign of dividend, so we use floor_divide instead.
    return x1 - x1._binary_op(x2, "floor_divide_op") * x2


def remainder(x1, x2, **kwargs):
    """Element-wise remainder of division (alias for mod)."""
    return mod(x1, x2)


# ------------------------------------------------------------------ NaN / comparison utilities

def isnan(x, **kwargs):
    """Test element-wise for NaN."""
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_predicate_op("isnan_op")


def isinf(x, **kwargs):
    """Test element-wise for positive or negative infinity."""
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_predicate_op("isinf_op")


def isfinite(x, **kwargs):
    """Test element-wise for finiteness."""
    if type(x) is not ndarray:
        x = creation.asarray(x)
    return x._unary_predicate_op("isfinite_op")


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """Replace NaN with zero and infinity with large finite numbers."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if type(x) is not ndarray:
        x = creation.asarray(x)

    if not np.issubdtype(x.dtype, np.floating):
        return x.copy() if copy else x

    # Default values for inf replacement
    if posinf is None:
        finfo = np.finfo(x.dtype)
        posinf = finfo.max
    if neginf is None:
        finfo = np.finfo(x.dtype)
        neginf = finfo.min

    # CPU fast path for small/medium arrays
    if x.size < 4194304:
        if x._np_data is None:
            MetalBackend().synchronize()
        result = ndarray._from_np_direct(
            np.nan_to_num(x._get_view(), copy=True, nan=nan, posinf=posinf, neginf=neginf))
        if not copy:
            x._np_data = result._np_data
            x._buffer = None
            return x
        return result

    metal_type = METAL_TYPE_NAMES[np.dtype(x.dtype)]
    shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void nan_to_num_op(device {metal_type} *a [[buffer(0)]],
                           device {metal_type} *params [[buffer(1)]],
                           device {metal_type} *out [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {{
    float v = static_cast<float>(a[id]);
    if (isnan(v)) out[id] = params[0];
    else if (isinf(v) && v > 0) out[id] = params[1];
    else if (isinf(v) && v < 0) out[id] = params[2];
    else out[id] = a[id];
}}
"""
    x_c = x._ensure_contiguous()
    backend = MetalBackend()
    params = np.array([nan, posinf, neginf], dtype=x.dtype)
    params_buf = backend.array_to_buffer(params)
    out_buf = backend.create_buffer(x_c.size, x.dtype)
    backend.execute_kernel(shader_src, "nan_to_num_op", x_c.size, [x_c._buffer, params_buf, out_buf])
    result = ndarray._from_buffer(out_buf, x.shape, x.dtype)
    if not copy:
        x._adopt_buffer(result._buffer)
        return x
    return result


def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Return boolean array where two arrays are element-wise equal within tolerance."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)

    # CPU fast path
    if a.size < 4194304 and b.size < 4194304:
        return ndarray._from_np_direct(np.isclose(a.get(), b.get(), rtol=rtol, atol=atol, equal_nan=equal_nan))

    if equal_nan:
        return ndarray._from_np_direct(np.isclose(a.get(), b.get(), rtol=rtol, atol=atol, equal_nan=equal_nan))

    # GPU path: |a - b| <= atol + rtol * |b|
    diff = a - b
    abs_diff = diff._unary_op("abs_op")
    abs_b = b._unary_op("abs_op")

    tolerance = creation.asarray(np.array(atol, dtype=np.float32)) + creation.asarray(np.array(rtol, dtype=np.float32)) * abs_b
    result = abs_diff._comparison_op(tolerance, "le_op")
    return result


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Return True if all elements are close within tolerance."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    # CPU fast path
    if a.size < 4194304 and b.size < 4194304:
        return bool(np.allclose(a.get(), b.get(), rtol=rtol, atol=atol, equal_nan=equal_nan))
    close = isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return bool(close.all())


def array_equal(a, b, equal_nan=False):
    """Return True if two arrays have the same shape and elements."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    if a.shape != b.shape:
        return False
    # CPU fast path
    if a.size < 4194304:
        return bool(np.array_equal(a.get(), b.get(), equal_nan=equal_nan))
    if equal_nan:
        return bool(np.array_equal(a.get(), b.get(), equal_nan=equal_nan))
    eq = a._comparison_op(b, "eq_op")
    return bool(eq.all())


def count_nonzero(a, axis=None, keepdims=False):
    """Count non-zero elements."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path
    if a.size < 4194304:
        np_data = a._np_data if a._np_data is not None else a.get()
        r = np.count_nonzero(np_data, axis=axis, keepdims=keepdims)
        if axis is None and not keepdims:
            return int(r)
        return ndarray._from_np_direct(np.asarray(r))

    # Use GPU comparison: a != 0
    zero = creation.zeros(a.shape, dtype=a.dtype)
    nonzero_mask = a._comparison_op(zero, "ne_op")  # returns bool array as int32

    # Sum the mask (cast to float32 for reduction)
    mask_float = nonzero_mask.astype(np.float32)

    if axis is None and not keepdims:
        result = mask_float.sum()
        return int(result.get())
    result = mask_float.sum(axis=axis, keepdims=keepdims)
    return result.astype(np.intp)


# ------------------------------------------------------------------ utility functions

def copy(a, order='K'):
    """Return a copy of the array."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # CPU fast path for CPU-resident arrays
    if a._np_data is not None:
        return ndarray._from_np_direct(a._get_view().copy())

    a_c = a._ensure_contiguous()
    metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
    shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void copy_op(device {metal_type} *src [[buffer(0)]],
                     device {metal_type} *dst [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    dst[id] = src[id];
}}
"""
    backend = MetalBackend()
    out_buf = backend.create_buffer(a_c.size, a.dtype)
    backend.execute_kernel(shader_src, "copy_op", a_c.size, [a_c._buffer, out_buf])
    return ndarray._from_buffer(out_buf, a.shape, a.dtype)


def ascontiguousarray(a):
    """Return a contiguous array in memory (C order)."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    # CPU fast path: just ensure C-contiguity via numpy
    if a._np_data is not None:
        if a._np_data.flags['C_CONTIGUOUS']:
            return a  # already contiguous, no copy needed
        return ndarray._from_np_direct(np.ascontiguousarray(a._np_data))
    return copy(a)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """Return the sum along diagonals of the array."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    if a.ndim < 2:
        result = np.trace(a.get(), offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        result_arr = ndarray._from_np_direct(result)
        if out is not None:
            out._np_data = result_arr.get()
            out._buffer = None
            return out
        return result_arr

    # For simple 2D case with offset=0 and default axes, use GPU
    if a.ndim == 2 and offset == 0 and axis1 == 0 and axis2 == 1:
        rows, cols = a.shape
        diag_len = min(rows, cols)
        a_c = a._ensure_contiguous()

        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void trace_op(device {metal_type} *a [[buffer(0)]],
                      device {metal_type} *out [[buffer(1)]],
                      device uint *dims [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {{
    if (id >= 1) return;
    uint cols = dims[0];
    uint diag_len = dims[1];
    {metal_type} sum = 0;
    for (uint i = 0; i < diag_len; i++) {{
        sum += a[i * cols + i];
    }}
    out[0] = sum;
}}
"""
        backend = MetalBackend()
        dims = np.array([cols, diag_len], dtype=np.uint32)
        dims_buf = backend.array_to_buffer(dims)
        out_buf = backend.create_buffer(1, a.dtype)
        backend.execute_kernel(shader_src, "trace_op", 1, [a_c._buffer, out_buf, dims_buf])
        result = ndarray._from_buffer(out_buf, (), a.dtype)
        if dtype is not None:
            result = result.astype(dtype)
        if out is not None:
            out._adopt_buffer(result._ensure_contiguous()._buffer)
            return out
        return result

    # Fall back for offset != 0 or non-standard axes
    result = np.trace(a.get(), offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    result_arr = ndarray._from_np_direct(result)
    if out is not None:
        out._np_data = result_arr.get()
        out._buffer = None
        return out
    return result_arr


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Return specified diagonals."""
    from ._metal_backend import MetalBackend
    from ._dtypes import METAL_TYPE_NAMES

    if not isinstance(a, ndarray):
        a = creation.asarray(a)

    # GPU path for simple 2D case
    if a.ndim == 2 and offset == 0 and axis1 == 0 and axis2 == 1:
        rows, cols = a.shape
        diag_len = min(rows, cols)
        a_c = a._ensure_contiguous()

        metal_type = METAL_TYPE_NAMES[np.dtype(a.dtype)]
        shader_src = f"""
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {{}}
kernel void diagonal_op(device {metal_type} *a [[buffer(0)]],
                         device {metal_type} *out [[buffer(1)]],
                         device uint *dims [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {{
    uint cols = dims[0];
    out[id] = a[id * cols + id];
}}
"""
        backend = MetalBackend()
        dims = np.array([cols], dtype=np.uint32)
        dims_buf = backend.array_to_buffer(dims)
        out_buf = backend.create_buffer(diag_len, a.dtype)
        backend.execute_kernel(shader_src, "diagonal_op", diag_len, [a_c._buffer, out_buf, dims_buf])
        return ndarray._from_buffer(out_buf, (diag_len,), a.dtype)

    # Fallback
    result = np.diagonal(a.get(), offset=offset, axis1=axis1, axis2=axis2)
    return ndarray._from_np_direct(np.ascontiguousarray(result))
