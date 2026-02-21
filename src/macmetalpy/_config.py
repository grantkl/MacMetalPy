"""Global configuration for macmetalpy."""

from __future__ import annotations

import threading
from typing import Literal

__all__ = ["MacMetalConfig", "get_config", "set_config"]


class MacMetalConfig:
    """Singleton configuration for macmetalpy behaviour.

    Attributes:
        float64_behavior: How to handle float64 inputs.
            ``"downcast"`` — silently convert to float32 (with optional warning).
            ``"cpu_fallback"`` — run the operation on CPU via NumPy.
        warn_on_downcast: Emit a ``UserWarning`` when float64 is downcast.
        default_float_dtype: Default float dtype for creation routines that
            receive no explicit dtype (mirrors CuPy's ``float32`` default).
    """

    _instance: MacMetalConfig | None = None
    _lock = threading.Lock()

    def __new__(cls) -> MacMetalConfig:
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst.float64_behavior: Literal["downcast", "cpu_fallback"] = "downcast"
                inst.warn_on_downcast: bool = True
                inst.default_float_dtype = "float32"
                cls._instance = inst
            return cls._instance


def get_config() -> MacMetalConfig:
    """Return the global macmetalpy configuration singleton."""
    return MacMetalConfig()


def set_config(
    *,
    float64_behavior: Literal["downcast", "cpu_fallback"] | None = None,
    warn_on_downcast: bool | None = None,
    default_float_dtype: str | None = None,
) -> None:
    """Update global macmetalpy configuration values."""
    cfg = get_config()
    if float64_behavior is not None:
        if float64_behavior not in ("downcast", "cpu_fallback"):
            raise ValueError(
                f"float64_behavior must be 'downcast' or 'cpu_fallback', "
                f"got {float64_behavior!r}"
            )
        cfg.float64_behavior = float64_behavior
    if warn_on_downcast is not None:
        cfg.warn_on_downcast = warn_on_downcast
    if default_float_dtype is not None:
        cfg.default_float_dtype = default_float_dtype
