"""Bitwise operations (CuPy-compatible)."""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from ._kernel_cache import KernelCache
from ._metal_backend import MetalBackend
from ._broadcasting import broadcast_shapes, needs_broadcast
from . import creation


def _ensure(x):
    return x if isinstance(x, ndarray) else creation.asarray(x)


def _cpu_view(a):
    """Get zero-copy numpy view (syncs if GPU-resident)."""
    if a._np_data is None:
        MetalBackend().synchronize()
    return a._get_view()


# ── Bitwise operations ───────────────────────────────────────────────


_BITWISE_NP = {
    "bit_and_op": np.bitwise_and, "bit_or_op": np.bitwise_or,
    "bit_xor_op": np.bitwise_xor, "lshift_op": np.left_shift,
    "rshift_op": np.right_shift,
}
_GPU_THRESHOLD_MEMORY = 4194304  # 4M — bitwise ops are pure memory, CPU SIMD always wins


def _binary_bitwise(x1, x2, kernel_name):
    x1, x2 = _ensure(x1), _ensure(x2)
    # CPU fallback — bitwise ops are pure memory operations
    if x1.size < _GPU_THRESHOLD_MEMORY and x2.size < _GPU_THRESHOLD_MEMORY:
        np_func = _BITWISE_NP.get(kernel_name)
        if np_func is not None:
            a_np = (x1._np_data if x1._np_data is not None else _cpu_view(x1)).astype(np.int32, copy=False)
            b_np = (x2._np_data if x2._np_data is not None else _cpu_view(x2)).astype(np.int32, copy=False)
            return ndarray._from_np_direct(np_func(a_np, b_np))
    backend = MetalBackend()
    cache = KernelCache()
    a = x1.astype(np.int32)._ensure_contiguous()
    b = x2.astype(np.int32)._ensure_contiguous()
    if needs_broadcast(a._shape, b._shape):
        out_shape = broadcast_shapes(a._shape, b._shape)
        a = creation.array(np.ascontiguousarray(np.broadcast_to(a.get(), out_shape)), dtype=np.int32)._ensure_contiguous()
        b = creation.array(np.ascontiguousarray(np.broadcast_to(b.get(), out_shape)), dtype=np.int32)._ensure_contiguous()
    shader = cache.get_shader("boolean", np.int32)
    out_buf = backend.create_buffer(a.size, np.int32)
    backend.execute_kernel(shader, kernel_name, a.size, [a._buffer, b._buffer, out_buf])
    return ndarray._from_buffer(out_buf, a._shape, np.int32)


def bitwise_and(x1, x2, **kwargs):
    return _binary_bitwise(x1, x2, "bit_and_op")


def bitwise_or(x1, x2, **kwargs):
    return _binary_bitwise(x1, x2, "bit_or_op")


def bitwise_xor(x1, x2, **kwargs):
    return _binary_bitwise(x1, x2, "bit_xor_op")


def invert(x, **kwargs):
    x = _ensure(x)
    # CPU fallback — pure memory op
    if x.size < _GPU_THRESHOLD_MEMORY:
        a_np = (x._np_data if x._np_data is not None else _cpu_view(x)).astype(np.int32, copy=False)
        return ndarray._from_np_direct(np.invert(a_np))
    backend = MetalBackend()
    cache = KernelCache()
    a = x.astype(np.int32)._ensure_contiguous()
    shader = cache.get_shader("boolean", np.int32)
    out_buf = backend.create_buffer(a.size, np.int32)
    backend.execute_kernel(shader, "bit_invert_op", a.size, [a._buffer, out_buf])
    return ndarray._from_buffer(out_buf, a._shape, np.int32)


bitwise_invert = invert
bitwise_not = invert


def left_shift(x1, x2, **kwargs):
    return _binary_bitwise(x1, x2, "lshift_op")


def right_shift(x1, x2, **kwargs):
    return _binary_bitwise(x1, x2, "rshift_op")


# ── Bit packing ───────────────────────────────────────────────────────


def packbits(a, axis=None):
    a = _ensure(a)
    result = np.packbits(_cpu_view(a).astype(np.uint8, copy=False), axis=axis)
    return ndarray._from_np_direct(result.astype(np.uint16))


def unpackbits(a, axis=None):
    a = _ensure(a)
    result = np.unpackbits(_cpu_view(a).astype(np.uint8, copy=False), axis=axis)
    return ndarray._from_np_direct(result.astype(np.uint16))


# ── Math ──────────────────────────────────────────────────────────────


def gcd(x1, x2, **kwargs):
    x1, x2 = _ensure(x1), _ensure(x2)
    return ndarray._from_np_direct(np.gcd(_cpu_view(x1), _cpu_view(x2)))


def lcm(x1, x2, **kwargs):
    x1, x2 = _ensure(x1), _ensure(x2)
    return ndarray._from_np_direct(np.lcm(_cpu_view(x1), _cpu_view(x2)))


def bitwise_count(a):
    """Count the number of 1-bits (popcount) in each element."""
    a = _ensure(a)
    a_np = a._np_data if a._np_data is not None else _cpu_view(a)
    result = np.bitwise_count(a_np)
    # bitwise_count returns uint8; convert to int32 for Metal compatibility
    if result.dtype == np.uint8:
        result = result.view(np.int8).astype(np.int16, copy=False)
    return ndarray._from_np_direct(result)
