"""NumPy dtype <-> Metal type mapping and dtype resolution."""

from __future__ import annotations

import warnings

import numpy as np

from ._config import get_config

__all__ = [
    "numpy_to_metal",
    "metal_to_numpy",
    "resolve_dtype",
    "result_dtype",
    "is_float_dtype",
    "SUPPORTED_DTYPES",
]

# Canonical mapping: NumPy dtype -> Metal type string
_NUMPY_TO_METAL: dict[np.dtype, str] = {
    np.dtype(np.float32): "float",
    np.dtype(np.float16): "half",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
    np.dtype(np.int64): "long",
    np.dtype(np.uint64): "uint64_t",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "uint16_t",
    np.dtype(np.bool_): "bool",
}

_METAL_TO_NUMPY: dict[str, np.dtype] = {v: k for k, v in _NUMPY_TO_METAL.items()}

SUPPORTED_DTYPES = frozenset(_NUMPY_TO_METAL.keys())

# Metal MSL type names used inside shader source
METAL_TYPE_NAMES: dict[np.dtype, str] = {
    np.dtype(np.float32): "float",
    np.dtype(np.float16): "half",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
    np.dtype(np.int64): "long",
    np.dtype(np.uint64): "unsigned long",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "unsigned short",
    np.dtype(np.bool_): "bool",
}


def numpy_to_metal(dtype: np.dtype) -> str:
    """Convert a NumPy dtype to its MetalGPU type string."""
    dtype = np.dtype(dtype)
    try:
        return _NUMPY_TO_METAL[dtype]
    except KeyError:
        raise TypeError(f"Unsupported dtype for Metal: {dtype}") from None


def metal_to_numpy(metal_type: str) -> np.dtype:
    """Convert a Metal type string to its NumPy dtype."""
    try:
        return _METAL_TO_NUMPY[metal_type]
    except KeyError:
        raise TypeError(f"Unknown Metal type: {metal_type!r}") from None


def is_float_dtype(dtype: np.dtype) -> bool:
    """Return True if *dtype* is a floating-point type."""
    return np.issubdtype(dtype, np.floating)


# Fast set of dtypes that pass through resolve_dtype unchanged
_PASSTHROUGH_DTYPES = frozenset(SUPPORTED_DTYPES | {np.dtype(np.complex64)})


def resolve_dtype(dtype) -> np.dtype:
    """Resolve a user-supplied dtype, handling float64 per config.

    Parameters
    ----------
    dtype : dtype-like or None
        If *None*, falls back to the configured default float dtype.

    Returns
    -------
    np.dtype
        A dtype supported by Metal.
    """
    # Fast path for already-supported dtypes (avoids get_config() call)
    if isinstance(dtype, np.dtype):
        if dtype in _PASSTHROUGH_DTYPES:
            return dtype
    elif dtype is not None:
        # Handle type objects like np.float32 without calling get_config()
        dtype = np.dtype(dtype)
        if dtype in _PASSTHROUGH_DTYPES:
            return dtype
    else:
        # dtype is None
        cfg = get_config()
        return np.dtype(cfg.default_float_dtype)

    cfg = get_config()

    # Handle float64 — Metal does not support it
    if dtype == np.float64:
        if cfg.float64_behavior == "downcast":
            if cfg.warn_on_downcast:
                warnings.warn(
                    "float64 is not supported on Metal; downcasting to float32.",
                    UserWarning,
                    stacklevel=3,
                )
            return np.dtype(cfg.default_float_dtype)
        # "cpu_fallback" — caller is responsible for actually running on CPU
        return dtype

    # Handle complex128 — downcast to complex64 (mirrors float64 handling)
    if dtype == np.complex128:
        return np.dtype(np.complex64)

    # Accept complex64 even though it has no Metal kernel type
    if dtype == np.complex64:
        return dtype

    # Handle int8/uint8 — upcast to int16/uint16 (Metal doesn't support 8-bit ints)
    if dtype == np.int8:
        return np.dtype(np.int16)
    if dtype == np.uint8:
        return np.dtype(np.uint16)

    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"Unsupported dtype for Metal: {dtype}")

    return dtype


def result_dtype(dt1: np.dtype, dt2: np.dtype) -> np.dtype:
    """Determine the result dtype for a binary operation using NumPy promotion.

    The promoted dtype is then resolved through :func:`resolve_dtype` so that
    float64 promotion is handled according to the global config.
    """
    promoted = np.result_type(dt1, dt2)
    return resolve_dtype(promoted)
