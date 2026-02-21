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


# ── Bitwise operations ───────────────────────────────────────────────


def _binary_bitwise(x1, x2, kernel_name):
    x1, x2 = _ensure(x1), _ensure(x2)
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


def bitwise_and(x1, x2):
    return _binary_bitwise(x1, x2, "bit_and_op")


def bitwise_or(x1, x2):
    return _binary_bitwise(x1, x2, "bit_or_op")


def bitwise_xor(x1, x2):
    return _binary_bitwise(x1, x2, "bit_xor_op")


def invert(x):
    x = _ensure(x)
    backend = MetalBackend()
    cache = KernelCache()
    a = x.astype(np.int32)._ensure_contiguous()
    shader = cache.get_shader("boolean", np.int32)
    out_buf = backend.create_buffer(a.size, np.int32)
    backend.execute_kernel(shader, "bit_invert_op", a.size, [a._buffer, out_buf])
    return ndarray._from_buffer(out_buf, a._shape, np.int32)


bitwise_invert = invert


def left_shift(x1, x2):
    return _binary_bitwise(x1, x2, "lshift_op")


def right_shift(x1, x2):
    return _binary_bitwise(x1, x2, "rshift_op")


# ── Bit packing ───────────────────────────────────────────────────────


def packbits(a, axis=None):
    a = _ensure(a)
    result = np.packbits(a.get().astype(np.uint8), axis=axis)
    return creation.array(result.astype(np.uint16))


def unpackbits(a, axis=None):
    a = _ensure(a)
    result = np.unpackbits(a.get().astype(np.uint8), axis=axis)
    return creation.array(result.astype(np.uint16))


# ── Math ──────────────────────────────────────────────────────────────


def gcd(x1, x2):
    x1, x2 = _ensure(x1), _ensure(x2)
    return creation.array(np.gcd(x1.get(), x2.get()))


def lcm(x1, x2):
    x1, x2 = _ensure(x1), _ensure(x2)
    return creation.array(np.lcm(x1.get(), x2.get()))
