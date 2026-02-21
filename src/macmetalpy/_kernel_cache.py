"""Caching layer for compiled Metal shader source strings."""

from __future__ import annotations

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
    "boolean": lambda metal_type: _kernels.boolean_shader(),
    "where": _kernels.where_shader,
    "clip": _kernels.clip_shader,
    "predicate": _kernels.predicate_shader,
    "axis_reduction": _kernels.axis_reduction_shader,
}


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
