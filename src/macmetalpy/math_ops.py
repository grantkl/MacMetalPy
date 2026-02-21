"""Module-level math functions (CuPy-compatible)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def sqrt(x):
    """Element-wise square root."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("sqrt_op")


def exp(x):
    """Element-wise exponential."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("exp_op")


def log(x):
    """Element-wise natural logarithm."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("log_op")


def abs(x):
    """Element-wise absolute value."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("abs_op")


def power(x1, x2):
    """Element-wise power."""
    if not isinstance(x1, ndarray):
        x1 = creation.asarray(x1)
    return x1._binary_op(x2, "pow_op")


def dot(a, b):
    """Dot product of two arrays."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    # 1-D dot product: use sum of elementwise multiply
    if a.ndim == 1 and b.ndim == 1:
        return (a * b).sum()
    # 2-D: use matmul
    return a.__matmul__(b)


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
    if not isinstance(x, ndarray):
        x = creation.asarray(x if not isinstance(x, (int, float)) else
                              np.array(x, dtype=np.float32))
    if not isinstance(y, ndarray):
        y = creation.asarray(y if not isinstance(y, (int, float)) else
                              np.array(y, dtype=np.float32))

    # Determine output dtype from x and y
    rdtype = result_dtype(x.dtype, y.dtype)
    x = x.astype(rdtype) if x.dtype != rdtype else x
    y = y.astype(rdtype) if y.dtype != rdtype else y

    # Broadcast all three to the same shape
    out_shape = broadcast_shapes(broadcast_shapes(condition.shape, x.shape), y.shape)
    if condition.shape != out_shape:
        condition = creation.array(
            np.ascontiguousarray(np.broadcast_to(condition.get(), out_shape)))
    if x.shape != out_shape:
        x = creation.array(
            np.ascontiguousarray(np.broadcast_to(x.get(), out_shape)), dtype=rdtype)
    if y.shape != out_shape:
        y = creation.array(
            np.ascontiguousarray(np.broadcast_to(y.get(), out_shape)), dtype=rdtype)

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


def clip(a, a_min, a_max):
    """Clip (limit) the values in an array (GPU-native)."""
    from ._kernel_cache import KernelCache
    from ._metal_backend import MetalBackend

    if not isinstance(a, ndarray):
        a = creation.asarray(a)

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
    return ndarray._from_buffer(out_buf, a.shape, a.dtype)


def concatenate(arrays, axis=0):
    """Join a sequence of arrays along an existing axis."""
    np_arrays = []
    for arr in arrays:
        if isinstance(arr, ndarray):
            np_arrays.append(arr.get())
        else:
            np_arrays.append(np.asarray(arr))
    result_np = np.concatenate(np_arrays, axis=axis)
    return creation.array(result_np)


def stack(arrays, axis=0):
    """Join a sequence of arrays along a new axis."""
    np_arrays = []
    for arr in arrays:
        if isinstance(arr, ndarray):
            np_arrays.append(arr.get())
        else:
            np_arrays.append(np.asarray(arr))
    result_np = np.stack(np_arrays, axis=axis)
    return creation.array(result_np)


def vstack(arrays):
    """Stack arrays vertically (row-wise)."""
    np_arrays = []
    for arr in arrays:
        if isinstance(arr, ndarray):
            np_arrays.append(arr.get())
        else:
            np_arrays.append(np.asarray(arr))
    result_np = np.vstack(np_arrays)
    return creation.array(result_np)


def hstack(arrays):
    """Stack arrays horizontally (column-wise)."""
    np_arrays = []
    for arr in arrays:
        if isinstance(arr, ndarray):
            np_arrays.append(arr.get())
        else:
            np_arrays.append(np.asarray(arr))
    result_np = np.hstack(np_arrays)
    return creation.array(result_np)


def sign(x):
    """Element-wise sign function."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("sign_op")


def floor(x):
    """Element-wise floor."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("floor_op")


def ceil(x):
    """Element-wise ceiling."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("ceil_op")


def sin(x):
    """Element-wise sine."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("sin_op")


def cos(x):
    """Element-wise cosine."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("cos_op")


def tan(x):
    """Element-wise tangent."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("tan_op")


def arcsin(x):
    """Element-wise inverse sine."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("asin_op")


def arccos(x):
    """Element-wise inverse cosine."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("acos_op")


def arctan(x):
    """Element-wise inverse tangent."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("atan_op")


def sinh(x):
    """Element-wise hyperbolic sine."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("sinh_op")


def cosh(x):
    """Element-wise hyperbolic cosine."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("cosh_op")


def tanh(x):
    """Element-wise hyperbolic tangent."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("tanh_op")


def log2(x):
    """Element-wise base-2 logarithm."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("log2_op")


def log10(x):
    """Element-wise base-10 logarithm."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("log10_op")


def square(x):
    """Element-wise square."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("square_op")


def negative(x):
    """Element-wise negation."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_op("negative_op")


def around(x, decimals=0):
    """Round to the given number of decimals."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    result_np = np.around(x.get(), decimals=decimals)
    return creation.array(result_np)


def round_(x, decimals=0):
    """Round to the given number of decimals (alias for around)."""
    return around(x, decimals=decimals)


round = around


def mod(x1, x2):
    """Element-wise remainder of division (NumPy-compatible: sign of divisor)."""
    if not isinstance(x1, ndarray):
        x1 = creation.asarray(x1)
    if not isinstance(x2, ndarray):
        x2 = creation.asarray(x2)
    # NumPy mod: a - floor(a/b)*b  (result has sign of divisor)
    # C fmod gives sign of dividend, so we use floor_divide instead.
    return x1 - x1._binary_op(x2, "floor_divide_op") * x2


def remainder(x1, x2):
    """Element-wise remainder of division (alias for mod)."""
    return mod(x1, x2)


# ------------------------------------------------------------------ NaN / comparison utilities

def isnan(x):
    """Test element-wise for NaN."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_predicate_op("isnan_op")


def isinf(x):
    """Test element-wise for positive or negative infinity."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_predicate_op("isinf_op")


def isfinite(x):
    """Test element-wise for finiteness."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    return x._unary_predicate_op("isfinite_op")


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """Replace NaN with zero and infinity with large finite numbers."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    result = np.nan_to_num(x.get(), nan=nan, posinf=posinf, neginf=neginf)
    return creation.array(result)


def isclose(a, b, rtol=1e-5, atol=1e-8):
    """Return boolean array where two arrays are element-wise equal within tolerance."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    return creation.array(np.isclose(a.get(), b.get(), rtol=rtol, atol=atol))


def allclose(a, b, rtol=1e-5, atol=1e-8):
    """Return True if all elements are close within tolerance."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    return bool(np.allclose(a.get(), b.get(), rtol=rtol, atol=atol))


def array_equal(a, b):
    """Return True if two arrays have the same shape and elements."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    return bool(np.array_equal(a.get(), b.get()))


def count_nonzero(a, axis=None):
    """Count non-zero elements."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.count_nonzero(a.get(), axis=axis)
    if axis is not None:
        return creation.array(np.asarray(result))
    return int(result)


# ------------------------------------------------------------------ utility functions

def copy(a):
    """Return a copy of the array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return creation.array(a.get())


def ascontiguousarray(a):
    """Return a contiguous array in memory (C order)."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return creation.array(a.get())


def trace(a, offset=0):
    """Return the sum along diagonals of the array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.trace(a.get(), offset=offset)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def diagonal(a, offset=0):
    """Return specified diagonals."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.diagonal(a.get(), offset=offset)
    return creation.array(result)
