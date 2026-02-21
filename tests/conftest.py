"""Shared test infrastructure mirroring CuPy's cupy.testing module.

Dtype groups, shape groups, tolerance helpers, and comparison utilities
that all test files import via conftest fixtures.

Ref: https://docs.cupy.dev/en/stable/reference/testing.html
"""

import itertools

import numpy as np
import numpy.testing as npt
import pytest

from macmetalpy._config import get_config

# ── CuPy-equivalent dtype groups ──────────────────────────────
# Mapped from CuPy's for_all_dtypes, for_float_dtypes, etc.
# minus types Metal doesn't support (float64, int8, uint8, complex128).

FLOAT_DTYPES = [np.float16, np.float32]
# CuPy: [float16, float32, float64]

SIGNED_DTYPES = [np.int16, np.int32, np.int64]
# CuPy: [int8, int16, int32, int64]

UNSIGNED_DTYPES = [np.uint16, np.uint32, np.uint64]
# CuPy: [uint8, uint16, uint32, uint64]

INT_DTYPES = SIGNED_DTYPES + UNSIGNED_DTYPES

COMPLEX_DTYPES = [np.complex64]
# CuPy: [complex64, complex128]

ALL_DTYPES = [np.bool_] + FLOAT_DTYPES + SIGNED_DTYPES + UNSIGNED_DTYPES + COMPLEX_DTYPES
# 10 types total (CuPy: 16)

ALL_DTYPES_NO_BOOL = [d for d in ALL_DTYPES if d != np.bool_]
ALL_DTYPES_NO_FLOAT16 = [d for d in ALL_DTYPES if d != np.float16]
ALL_DTYPES_NO_COMPLEX = [d for d in ALL_DTYPES if d != np.complex64]
NUMERIC_DTYPES = FLOAT_DTYPES + INT_DTYPES  # 8 types

# ── Dtype pairs for binary op combination tests ───────────────
# CuPy's for_all_dtypes_combination tests every N×N pair.
DTYPE_PAIRS = list(itertools.product(ALL_DTYPES_NO_BOOL, repeat=2))  # 81
DTYPE_PAIRS_WITH_BOOL = list(itertools.product(ALL_DTYPES, repeat=2))  # 100

# ── Shape groups (from CuPy's arithmetic/elementwise tests) ───
SHAPES_SCALAR = [()]
SHAPES_1D = [(1,), (5,), (1024,)]
SHAPES_2D = [(2, 3), (3, 2), (1, 1)]
SHAPES_3D = [(2, 3, 4)]
SHAPES_ZERO = [(0,), (3, 0, 2)]
SHAPES_ALL = SHAPES_SCALAR + SHAPES_1D + SHAPES_2D + SHAPES_3D + SHAPES_ZERO  # 10
SHAPES_NONZERO = SHAPES_1D + SHAPES_2D + SHAPES_3D  # 7

# Broadcast pairs (from CuPy's test_ndarray_elementwise_op.py)
BROADCAST_PAIRS = [
    ((2, 3), (3,)),
    ((2, 3), (2, 1)),
    ((2, 1, 3), (3, 1)),
    ((1,), (2, 3)),
]


# ── Tolerance helpers (matching CuPy's numpy_cupy_allclose) ───
def tol_for(dtype, category="default"):
    """Return dict(rtol=, atol=) matching CuPy's tolerance patterns.

    Categories mirror CuPy's per-test-group tolerances:
    - "default":     rtol=1e-5 for float32, exact for int/bool
    - "arithmetic":  CuPy ArithmeticBinaryBase atol=1e-4
    - "unary_math":  CuPy TestArithmeticUnary atol=1e-5
    - "power":       CuPy power tests atol=1.0
    """
    dtype = np.dtype(dtype)

    if category == "arithmetic":
        if dtype == np.float16:
            return dict(rtol=1e-2, atol=1e-2)
        return dict(rtol=1e-6, atol=1e-4)

    if category == "unary_math":
        if dtype == np.float16:
            return dict(rtol=1e-2, atol=1e-2)
        return dict(rtol=1e-5, atol=1e-5)

    if category == "power":
        return dict(rtol=1e-6, atol=1.0)

    # Default
    if dtype == np.float16:
        return dict(rtol=1e-2, atol=1e-3)
    if dtype in (np.float32, np.complex64):
        return dict(rtol=1e-5, atol=1e-6)
    return dict(rtol=0, atol=0)


# ── Comparison helpers (mirror CuPy's @numpy_cupy_allclose) ───
def assert_eq(gpu_arr, np_ref, dtype=None, category="default"):
    """Compare GPU array against NumPy reference with dtype-aware tolerance."""
    result = gpu_arr.get() if hasattr(gpu_arr, "get") else np.asarray(gpu_arr)
    ref = np.asarray(np_ref)
    d = dtype or ref.dtype
    npt.assert_allclose(result, ref, **tol_for(d, category))


def assert_shape_dtype(gpu_arr, expected_shape, expected_dtype):
    """Verify shape and dtype without checking values."""
    assert gpu_arr.shape == expected_shape
    assert gpu_arr.dtype == np.dtype(expected_dtype)


def make_arg(shape, dtype, xp=np):
    """Create a deterministic test array for the given shape and dtype.

    Uses integers to avoid precision issues across dtypes.
    """
    if not shape:
        return xp.array(1, dtype=dtype)
    size = 1
    for s in shape:
        size *= s
    if size == 0:
        return xp.zeros(shape, dtype=dtype)
    data = np.arange(1, size + 1, dtype=np.float64).reshape(shape)
    if np.issubdtype(dtype, np.bool_):
        return xp.array(data % 2 == 0, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return xp.array((data + 1j * data).astype(dtype))
    return xp.array(data.astype(dtype))


# ── pytest fixtures ───────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_config():
    """Reset the config singleton to defaults before each test."""
    cfg = get_config()
    cfg.float64_behavior = "downcast"
    cfg.warn_on_downcast = True
    cfg.default_float_dtype = "float32"
    yield


@pytest.fixture(params=ALL_DTYPES, ids=lambda d: np.dtype(d).name)
def dtype(request):
    return request.param


@pytest.fixture(params=ALL_DTYPES_NO_BOOL, ids=lambda d: np.dtype(d).name)
def dtype_no_bool(request):
    return request.param


@pytest.fixture(params=FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
def float_dtype(request):
    return request.param


@pytest.fixture(params=INT_DTYPES, ids=lambda d: np.dtype(d).name)
def int_dtype(request):
    return request.param


@pytest.fixture(params=SIGNED_DTYPES, ids=lambda d: np.dtype(d).name)
def signed_dtype(request):
    return request.param


@pytest.fixture(params=UNSIGNED_DTYPES, ids=lambda d: np.dtype(d).name)
def unsigned_dtype(request):
    return request.param


@pytest.fixture(params=NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
def numeric_dtype(request):
    return request.param


@pytest.fixture(params=SHAPES_ALL, ids=str)
def shape(request):
    return request.param


@pytest.fixture(params=SHAPES_NONZERO, ids=str)
def nonzero_shape(request):
    return request.param


@pytest.fixture(params=SHAPES_1D, ids=str)
def shape_1d(request):
    return request.param


@pytest.fixture(params=SHAPES_2D, ids=str)
def shape_2d(request):
    return request.param


@pytest.fixture(params=BROADCAST_PAIRS, ids=lambda p: f"{p[0]}+{p[1]}")
def broadcast_pair(request):
    return request.param
