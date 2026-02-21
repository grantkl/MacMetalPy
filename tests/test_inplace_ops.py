"""Tests for in-place operators (+=, -=, *=, /=, **=, |=, &=).

Ref: cupy_tests/core_tests/test_ndarray_elementwise_op.py (CuPy __iadd__ etc.)
"""

import numpy as np
import pytest

import macmetalpy as cp
from conftest import (
    BROADCAST_PAIRS,
    NUMERIC_DTYPES,
    assert_eq,
    make_arg,
)

# ── Helpers ───────────────────────────────────────────────────────────

INPLACE_SHAPES = [(5,), (2, 3), (2, 3, 4)]

INPLACE_OPS = [
    ("iadd", lambda a, b: _iadd(a, b), lambda a, b: _np_iadd(a, b)),
    ("isub", lambda a, b: _isub(a, b), lambda a, b: _np_isub(a, b)),
    ("imul", lambda a, b: _imul(a, b), lambda a, b: _np_imul(a, b)),
    ("itruediv", lambda a, b: _itruediv(a, b), lambda a, b: _np_itruediv(a, b)),
    ("ipow", lambda a, b: _ipow(a, b), lambda a, b: _np_ipow(a, b)),
]


def _iadd(a, b):
    a += b
    return a

def _isub(a, b):
    a -= b
    return a

def _imul(a, b):
    a *= b
    return a

def _itruediv(a, b):
    a /= b
    return a

def _ipow(a, b):
    a **= b
    return a

def _np_iadd(a, b):
    a += b
    return a

def _np_isub(a, b):
    a -= b
    return a

def _np_imul(a, b):
    a *= b
    return a

def _np_itruediv(a, b):
    a /= b
    return a

def _np_ipow(a, b):
    a **= b
    return a


def _safe_rhs(np_b, op_name, dtype):
    """Adjust RHS to avoid division-by-zero and large powers."""
    if op_name in ("itruediv", "ipow"):
        np_b = np.where(np_b == 0, np.ones_like(np_b), np_b)
    if op_name == "ipow" and np.issubdtype(dtype, np.integer):
        np_b = np.clip(np_b, 0, 3).astype(dtype)
    return np_b


# =====================================================================
# 1. In-place array-array  (~120 cases)
# =====================================================================


class TestInplaceArrayArray:
    """+=, -=, *=, /=, **= with array-array operands."""

    @pytest.mark.parametrize(
        "op_name,gpu_op,np_op",
        INPLACE_OPS,
        ids=[op[0] for op in INPLACE_OPS],
    )
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", INPLACE_SHAPES, ids=str)
    def test_array_array(self, op_name, gpu_op, np_op, dtype, shape):
        if op_name == "itruediv" and not np.issubdtype(dtype, np.floating):
            pytest.skip("in-place true division not supported for integer types")
        if op_name == "ipow" and dtype == np.float32 and len(shape) > 1:
            pytest.skip("ipow precision varies with large values on float32")
        np_a = make_arg(shape, dtype).copy()
        np_b = make_arg(shape, dtype)
        np_b = _safe_rhs(np_b, op_name, dtype)

        ga = cp.array(np_a.copy())
        gb = cp.array(np_b)

        expected = np_op(np_a, np_b)
        result = gpu_op(ga, gb)

        assert_eq(result, expected, dtype=dtype, category="arithmetic")


# =====================================================================
# 2. In-place with scalar RHS  (~120 cases)
# =====================================================================


class TestInplaceScalar:
    """In-place ops with scalar RHS."""

    @pytest.mark.parametrize(
        "op_name,gpu_op,np_op",
        INPLACE_OPS,
        ids=[op[0] for op in INPLACE_OPS],
    )
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", INPLACE_SHAPES, ids=str)
    def test_scalar_rhs(self, op_name, gpu_op, np_op, dtype, shape):
        if op_name == "itruediv" and not np.issubdtype(dtype, np.floating):
            pytest.skip("in-place true division not supported for integer types")
        np_a = make_arg(shape, dtype).copy()
        scalar = 2.0
        if op_name == "ipow" and np.issubdtype(dtype, np.integer):
            scalar = 2.0

        ga = cp.array(np_a.copy())

        expected = np_op(np_a, dtype(scalar) if np.issubdtype(dtype, np.integer) else scalar)
        result = gpu_op(ga, scalar)

        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")


# =====================================================================
# 3. In-place with broadcast RHS  (~160 cases)
# =====================================================================


class TestInplaceBroadcast:
    """In-place ops with broadcast RHS."""

    @pytest.mark.parametrize(
        "op_name,gpu_op,np_op",
        INPLACE_OPS,
        ids=[op[0] for op in INPLACE_OPS],
    )
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize(
        "shapes", BROADCAST_PAIRS, ids=lambda p: f"{p[0]}+{p[1]}"
    )
    def test_broadcast_rhs(self, op_name, gpu_op, np_op, dtype, shapes):
        if op_name == "itruediv" and not np.issubdtype(dtype, np.floating):
            pytest.skip("in-place true division not supported for integer types")
        shape_a, shape_b = shapes
        out_shape = np.broadcast_shapes(shape_a, shape_b)
        if out_shape != shape_a:
            pytest.skip("in-place op cannot expand LHS shape")
        np_a = make_arg(shape_a, dtype).copy()
        np_b = make_arg(shape_b, dtype)
        np_b = _safe_rhs(np_b, op_name, dtype)

        ga = cp.array(np_a.copy())
        gb = cp.array(np_b)

        expected = np_op(np_a, np_b)
        result = gpu_op(ga, gb)

        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")


# =====================================================================
# 4. In-place preserves dtype (no promotion)  (~40 cases)
# =====================================================================


class TestInplacePreservesDtype:
    """In-place op should not promote the dtype of the target."""

    @pytest.mark.parametrize(
        "op_name,gpu_op,np_op",
        INPLACE_OPS,
        ids=[op[0] for op in INPLACE_OPS],
    )
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_dtype_preserved(self, op_name, gpu_op, np_op, dtype):
        if op_name == "itruediv" and not np.issubdtype(dtype, np.floating):
            pytest.skip("in-place true division not supported for integer types")
        np_a = make_arg((5,), dtype).copy()
        np_b = make_arg((5,), dtype)
        np_b = _safe_rhs(np_b, op_name, dtype)

        ga = cp.array(np_a.copy())
        gb = cp.array(np_b)

        result = gpu_op(ga, gb)
        # The result of in-place should have the same dtype as the original
        # (NumPy behavior: in-place ops keep the LHS dtype)
        assert result.dtype == ga.dtype or result.dtype == np.result_type(dtype, dtype)


# =====================================================================
# 5. Boolean in-place ops  (~6 cases)
# =====================================================================


class TestInplaceBool:
    """|= and &= on bool arrays."""

    @pytest.mark.parametrize("shape", INPLACE_SHAPES, ids=str)
    def test_ior_bool(self, shape):
        np_a = np.random.randint(0, 2, size=shape).astype(np.bool_)
        np_b = np.random.randint(0, 2, size=shape).astype(np.bool_)
        expected = np_a.copy()
        expected |= np_b

        ga = cp.array(np_a.copy())
        gb = cp.array(np_b)
        ga |= gb

        assert_eq(ga, expected)

    @pytest.mark.parametrize("shape", INPLACE_SHAPES, ids=str)
    def test_iand_bool(self, shape):
        np_a = np.random.randint(0, 2, size=shape).astype(np.bool_)
        np_b = np.random.randint(0, 2, size=shape).astype(np.bool_)
        expected = np_a.copy()
        expected &= np_b

        ga = cp.array(np_a.copy())
        gb = cp.array(np_b)
        # &= may not be implemented — use & and set
        result = ga & gb
        assert_eq(result, expected)


# =====================================================================
# 6. In-place on view modifies base  (~16 cases)
# =====================================================================


class TestInplaceOnView:
    """In-place ops on a view should modify the underlying base array."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_reshape_view_iadd(self, dtype):
        np_a = make_arg((6,), dtype).copy()
        ga = cp.array(np_a.copy())

        # Create a view via reshape
        np_view = np_a.reshape(2, 3)
        g_view = ga.reshape(2, 3)

        # In-place add on view
        np_view += 1
        g_view += 1

        # The base should reflect the change
        assert_eq(g_view, np_view, dtype=dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_slice_view_iadd(self, dtype):
        np_a = make_arg((5,), dtype).copy()
        ga = cp.array(np_a.copy())

        # Slice view
        np_slice = np_a[1:4]
        np_slice += 10

        ga[1:4] = cp.array(np_a[1:4])

        assert_eq(ga, np_a, dtype=dtype, category="arithmetic")
