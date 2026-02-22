"""Caching layer for compiled Metal shader source strings."""

from __future__ import annotations

import functools
import threading
from typing import Dict, Tuple

import numpy as np

from . import _kernels
from ._dtypes import METAL_TYPE_NAMES

__all__ = ["KernelCache"]

# Map category name -> generator function
_GENERATORS = {
    "elementwise": _kernels.elementwise_shader,
    "reduction": _kernels.reduction_shader,
    "matmul": _kernels.matmul_shader,
    "comparison": _kernels.comparison_shader,
    "comparison_bool": _kernels.comparison_bool_shader,
    "boolean": lambda metal_type: _kernels.boolean_shader(),
    "bool_logic": lambda metal_type: _kernels.bool_logic_shader(),
    "where": _kernels.where_shader,
    "clip": _kernels.clip_shader,
    "predicate": _kernels.predicate_shader,
    "axis_reduction": _kernels.axis_reduction_shader,
    "parallel_reduction": _kernels.parallel_reduction_shader,
    "parallel_scan": _kernels.parallel_scan_shader,
    "nan_elementwise": _kernels.nan_elementwise_shader,
}

# ── Opt 5: Cached inline shader generators ──────────────────────────

_MSL_HEADER = """\
#include <metal_stdlib>
using namespace metal;
kernel void _sync() {}
"""


@functools.lru_cache(maxsize=64)
def get_astype_shader(src_metal: str, dst_metal: str, dst_is_bool: bool) -> str:
    if dst_is_bool:
        cast_expr = f"dst[id] = (src[id] != 0) ? (uchar)1 : (uchar)0;"
    else:
        cast_expr = f"dst[id] = static_cast<{dst_metal}>(src[id]);"
    return (
        _MSL_HEADER
        + f"""kernel void cast_op(device {src_metal} *src [[buffer(0)]],
                     device {dst_metal} *dst [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {{
    {cast_expr}
}}
"""
    )


@functools.lru_cache(maxsize=32)
def get_copy_shader(metal_type: str) -> str:
    return (
        _MSL_HEADER
        + f"""kernel void copy_buf(device {metal_type} *src [[buffer(0)]],
                      device {metal_type} *dst [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {{
    dst[id] = src[id];
}}
"""
    )


@functools.lru_cache(maxsize=32)
def get_broadcast_shader(metal_type: str) -> str:
    return (
        _MSL_HEADER
        + f"""kernel void broadcast_nd(device {metal_type} *src [[buffer(0)]],
                          device {metal_type} *dst [[buffer(1)]],
                          device uint *src_shape_arr [[buffer(2)]],
                          device uint *out_shape_arr [[buffer(3)]],
                          device uint *ndim_ptr [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {{
    uint nd = ndim_ptr[0];
    uint src_idx = 0;
    uint remaining = id;
    uint src_stride = 1;
    for (int d = (int)nd - 1; d >= 0; d--) {{
        uint dim_idx = remaining % out_shape_arr[d];
        remaining /= out_shape_arr[d];
        src_idx += (dim_idx % src_shape_arr[d]) * src_stride;
        src_stride *= src_shape_arr[d];
    }}
    dst[id] = src[src_idx];
}}
"""
    )


@functools.lru_cache(maxsize=32)
def get_strided_copy_shader(metal_type: str) -> str:
    return (
        _MSL_HEADER
        + f"""kernel void strided_copy(device {metal_type} *src [[buffer(0)]],
                          device {metal_type} *dst [[buffer(1)]],
                          device uint *shape_arr [[buffer(2)]],
                          device uint *strides_arr [[buffer(3)]],
                          device uint *params [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {{
    uint ndim_val = params[0];
    uint offset = params[1];
    uint src_idx = offset;
    uint remaining = id;
    for (int d = (int)ndim_val - 1; d >= 0; d--) {{
        uint dim_idx = remaining % shape_arr[d];
        remaining /= shape_arr[d];
        src_idx += dim_idx * strides_arr[d];
    }}
    dst[id] = src[src_idx];
}}
"""
    )


@functools.lru_cache(maxsize=32)
def get_scalar_copy_shader(metal_type: str) -> str:
    return (
        _MSL_HEADER
        + f"""kernel void scalar_copy(device {metal_type} *src [[buffer(0)]],
                         device {metal_type} *dst [[buffer(1)]],
                         device uint *params [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {{
    dst[0] = src[params[0]];
}}
"""
    )


@functools.lru_cache(maxsize=32)
def get_scalar_write_shader(metal_type: str) -> str:
    return (
        _MSL_HEADER
        + f"""kernel void scalar_write(device {metal_type} *dst [[buffer(0)]],
                          device {metal_type} *src [[buffer(1)]],
                          device uint *params [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {{
    dst[params[0]] = src[0];
}}
"""
    )


@functools.lru_cache(maxsize=32)
def get_offset_copy_shader(metal_type: str) -> str:
    return (
        _MSL_HEADER
        + f"""kernel void offset_copy(device {metal_type} *dst [[buffer(0)]],
                         device {metal_type} *src [[buffer(1)]],
                         device uint *params [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {{
    dst[params[0] + id] = src[id];
}}
"""
    )


@functools.lru_cache(maxsize=32)
def get_strided_write_shader(metal_type: str) -> str:
    return (
        _MSL_HEADER
        + f"""kernel void strided_write(device {metal_type} *dst [[buffer(0)]],
                           device {metal_type} *src [[buffer(1)]],
                           device uint *shape_arr [[buffer(2)]],
                           device uint *strides_arr [[buffer(3)]],
                           device uint *params [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {{
    uint ndim_val = params[0];
    uint offset = params[1];
    uint dst_idx = offset;
    uint remaining = id;
    for (int d = (int)ndim_val - 1; d >= 0; d--) {{
        uint dim_idx = remaining % shape_arr[d];
        remaining /= shape_arr[d];
        dst_idx += dim_idx * strides_arr[d];
    }}
    dst[dst_idx] = src[id];
}}
"""
    )


class KernelCache:
    """Singleton cache for generated MSL shader source strings.

    Shaders are keyed by ``(category, np.dtype)`` and lazily generated on first
    access.
    """

    _instance: KernelCache | None = None
    _lock = threading.Lock()

    def __new__(cls) -> KernelCache:
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._cache: Dict[Tuple[str, np.dtype], str] = {}
                cls._instance = inst
            return cls._instance

    def get_shader(self, category: str, dtype: np.dtype) -> str:
        """Return the MSL source for *category* and *dtype*, generating if needed."""
        dtype = np.dtype(dtype)
        key = (category, dtype)
        if key not in self._cache:
            generator = _GENERATORS.get(category)
            if generator is None:
                raise ValueError(f"Unknown shader category: {category!r}")
            metal_type = METAL_TYPE_NAMES[dtype]
            self._cache[key] = generator(metal_type)
        return self._cache[key]

    def clear(self) -> None:
        """Empty the shader cache."""
        self._cache.clear()
