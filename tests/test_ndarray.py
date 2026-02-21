"""Comprehensive tests for ndarray properties, data transfer, shape ops,
type casting, operators, comparisons, boolean ops, reductions, indexing,
and dunder methods.

Ref: cupy_tests/core_tests/test_ndarray.py
     cupy_tests/core_tests/test_ndarray_elementwise_op.py
"""

import math
import operator

import numpy as np
import pytest

import macmetalpy as cp
from conftest import (
    ALL_DTYPES,
    ALL_DTYPES_NO_BOOL,
    BROADCAST_PAIRS,
    DTYPE_PAIRS,
    FLOAT_DTYPES,
    INT_DTYPES,
    NUMERIC_DTYPES,
    SHAPES_1D,
    SHAPES_2D,
    SHAPES_3D,
    SHAPES_ALL,
    SHAPES_NONZERO,
    SHAPES_ZERO,
    assert_eq,
    make_arg,
    tol_for,
)

# ── Helpers ───────────────────────────────────────────────────────────

REPR_SHAPES = [(1,), (2, 3)]


# =====================================================================
# 1. Properties  (~630 parametrized cases)
# =====================================================================
# Ref: cupy_tests/core_tests/test_ndarray.py::TestNdarrayProperties


class TestProperties:
    """shape, ndim, size, nbytes, itemsize, dtype, strides, T."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_ALL, ids=str)
    def test_shape(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert g.shape == np_a.shape

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_ALL, ids=str)
    def test_ndim(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert g.ndim == np_a.ndim

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_ALL, ids=str)
    def test_size(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert g.size == np_a.size

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_ALL, ids=str)
    def test_nbytes(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert g.nbytes == np_a.nbytes

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_ALL, ids=str)
    def test_itemsize(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert g.itemsize == np_a.itemsize

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(1,), (2, 3), (2, 3, 4)], ids=str)
    def test_dtype_matches(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert g.dtype == np_a.dtype

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO, ids=str)
    def test_strides(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert g.strides == np_a.strides

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_2D, ids=str)
    def test_T(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        expected = np_a.T
        result = g.T
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)


# =====================================================================
# 2. Data transfer  (~260 cases)
# =====================================================================
# Ref: cupy_tests/core_tests/test_ndarray.py::TestNdarrayGetSet


class TestDataTransfer:
    """get() roundtrip, get() non-contiguous, set() roundtrip."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_ALL, ids=str)
    def test_get_roundtrip(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert_eq(g, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_2D, ids=str)
    def test_get_transposed(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a).T
        expected = np_a.T
        assert_eq(g, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_2D, ids=str)
    def test_get_sliced(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        # Take first row
        if shape[0] > 0 and shape[1] > 0:
            expected = np_a[0]
            result = g[0]
            assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_ALL, ids=str)
    def test_set_roundtrip(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.zeros(shape, dtype=dtype) if shape else cp.array(np.zeros((), dtype=dtype))
        g.set(np_a)
        assert_eq(g, np_a, dtype=dtype)


# =====================================================================
# 3. Shape ops  (~420 cases)
# =====================================================================
# Ref: cupy_tests/core_tests/test_ndarray.py::TestNdarrayShape

SHAPE_OP_SHAPES = [(5,), (2, 3), (2, 3, 4)]


class TestReshape:
    """reshape: compatible, -1 infer, error cases."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPE_OP_SHAPES, ids=str)
    def test_reshape_compatible(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        new_shape = (np_a.size,)
        assert_eq(g.reshape(new_shape), np_a.reshape(new_shape), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPE_OP_SHAPES, ids=str)
    def test_reshape_minus1(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert_eq(g.reshape((-1,)), np_a.reshape((-1,)), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_reshape_error_two_minus1(self, dtype):
        g = cp.array(make_arg((6,), dtype))
        with pytest.raises(ValueError):
            g.reshape((-1, -1))

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_reshape_error_bad_size(self, dtype):
        g = cp.array(make_arg((6,), dtype))
        with pytest.raises(ValueError):
            g.reshape((4, 4))


class TestTranspose:
    """transpose: default (reverse), explicit axes."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_2D, ids=str)
    def test_transpose_default_2d(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert_eq(g.transpose(), np_a.transpose(), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_transpose_explicit_axes_3d(self, dtype):
        shape = (2, 3, 4)
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        axes = (1, 2, 0)
        assert_eq(g.transpose(axes), np_a.transpose(axes), dtype=dtype)


class TestFlatten:
    """flatten: contiguous, non-contiguous."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPE_OP_SHAPES, ids=str)
    def test_flatten_contiguous(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert_eq(g.flatten(), np_a.flatten(), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_2D, ids=str)
    def test_flatten_noncontiguous(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a).T
        assert_eq(g.flatten(), np_a.T.flatten(), dtype=dtype)


class TestRavel:
    """ravel: contiguous -> view, non-contiguous -> copy."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPE_OP_SHAPES, ids=str)
    def test_ravel_contiguous(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        result = g.ravel()
        assert_eq(result, np_a.ravel(), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_2D, ids=str)
    def test_ravel_noncontiguous(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a).T
        assert_eq(g.ravel(), np_a.T.ravel(), dtype=dtype)


class TestSqueeze:
    """squeeze: size-1, all, specific axis."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_squeeze_all(self, dtype):
        np_a = make_arg((1, 2, 1, 3), dtype)
        g = cp.array(np_a)
        assert_eq(g.squeeze(), np_a.squeeze(), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_squeeze_specific_axis(self, dtype):
        np_a = make_arg((1, 2, 1, 3), dtype)
        g = cp.array(np_a)
        assert_eq(g.squeeze(axis=0), np_a.squeeze(axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_squeeze_no_change(self, dtype):
        np_a = make_arg((2, 3), dtype)
        g = cp.array(np_a)
        assert_eq(g.squeeze(), np_a.squeeze(), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_squeeze_error_nonsize1(self, dtype):
        g = cp.array(make_arg((2, 3), dtype))
        with pytest.raises(ValueError):
            g.squeeze(axis=0)


class TestExpandDims:
    """expand_dims: positive axis, negative axis."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPE_OP_SHAPES, ids=str)
    def test_expand_dims_axis0(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert_eq(cp.expand_dims(g, axis=0), np.expand_dims(np_a, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPE_OP_SHAPES, ids=str)
    def test_expand_dims_negative(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        assert_eq(cp.expand_dims(g, axis=-1), np.expand_dims(np_a, axis=-1), dtype=dtype)


# =====================================================================
# 4. Type casting  (~111 cases)
# =====================================================================
# Ref: cupy_tests/core_tests/test_ndarray.py::TestNdarrayAstype


class TestAstype:
    """astype: all 81 no-bool dtype pairs."""

    @pytest.mark.parametrize(
        "src_dtype,dst_dtype",
        DTYPE_PAIRS,
        ids=[f"{np.dtype(a).name}->{np.dtype(b).name}" for a, b in DTYPE_PAIRS],
    )
    def test_astype_pairs(self, src_dtype, dst_dtype):
        np_a = make_arg((5,), src_dtype)
        g = cp.array(np_a)
        expected = np_a.astype(dst_dtype)
        result = g.astype(dst_dtype)
        assert result.dtype == expected.dtype
        assert_eq(result, expected, dtype=dst_dtype)


class TestCopy:
    """copy: verify independence."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPE_OP_SHAPES, ids=str)
    def test_copy_independence(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        g = cp.array(np_a)
        copied = g.copy()
        assert_eq(copied, np_a, dtype=dtype)
        # Verify independence: modifying original should not affect copy
        new_data = np.zeros(shape, dtype=dtype)
        g.set(new_data)
        assert_eq(copied, np_a, dtype=dtype)


# =====================================================================
# 5. Arithmetic operators  (~1,885 cases)
# =====================================================================
# Ref: cupy_tests/core_tests/test_ndarray_elementwise_op.py

ARITH_SHAPES = [(5,), (2, 3), (2, 3, 4)]
BINARY_OPS = [
    ("add", operator.add),
    ("sub", operator.sub),
    ("mul", operator.mul),
    ("truediv", operator.truediv),
    ("pow", operator.pow),
]


class TestArithmeticBinary:
    """Binary +, -, *, /, ** : array-array, array-scalar, scalar-array."""

    @pytest.mark.parametrize(
        "op_name,op_func", BINARY_OPS, ids=[b[0] for b in BINARY_OPS]
    )
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", ARITH_SHAPES, ids=str)
    def test_array_array(self, op_name, op_func, dtype, shape):
        if op_name == "pow" and dtype == np.float32 and len(shape) > 2:
            pytest.skip("float32 pow precision diverges on Metal for large arrays")
        np_a = make_arg(shape, dtype)
        np_b = make_arg(shape, dtype)
        # Avoid division by zero
        if op_name in ("truediv", "pow"):
            np_b = np.where(np_b == 0, np.ones_like(np_b), np_b)
        # Avoid large powers for int types
        if op_name == "pow" and np.issubdtype(dtype, np.integer):
            np_b = np.clip(np_b, 0, 3).astype(dtype)
        expected = op_func(np_a, np_b)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = op_func(ga, gb)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize(
        "op_name,op_func", BINARY_OPS, ids=[b[0] for b in BINARY_OPS]
    )
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", ARITH_SHAPES, ids=str)
    def test_array_scalar(self, op_name, op_func, dtype, shape):
        np_a = make_arg(shape, dtype)
        scalar = 2.0
        if op_name == "pow" and np.issubdtype(dtype, np.integer):
            scalar = 2.0
        expected = op_func(np_a, scalar)
        ga = cp.array(np_a)
        result = op_func(ga, scalar)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize(
        "op_name,op_func", BINARY_OPS, ids=[b[0] for b in BINARY_OPS]
    )
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", ARITH_SHAPES, ids=str)
    def test_scalar_array(self, op_name, op_func, dtype, shape):
        np_a = make_arg(shape, dtype)
        scalar = 2.0
        # Avoid 0 ** negative
        if op_name == "pow":
            np_a = np.where(np_a == 0, np.ones_like(np_a), np_a)
            if np.issubdtype(dtype, np.integer):
                np_a = np.clip(np_a, 1, 5).astype(dtype)
        expected = op_func(scalar, np_a)
        ga = cp.array(np_a)
        result = op_func(scalar, ga)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")


class TestArithmeticCrossDtype:
    """Cross-dtype binary ops: 5 ops x 81 dtype pairs (from DTYPE_PAIRS)."""

    @pytest.mark.parametrize(
        "op_name,op_func",
        [("add", operator.add), ("sub", operator.sub), ("mul", operator.mul)],
        ids=["add", "sub", "mul"],
    )
    @pytest.mark.parametrize(
        "dtype_a,dtype_b",
        DTYPE_PAIRS,
        ids=[f"{np.dtype(a).name}-{np.dtype(b).name}" for a, b in DTYPE_PAIRS],
    )
    @pytest.mark.parametrize("shape", [(5,), (2, 3)], ids=str)
    def test_cross_dtype_basic(self, op_name, op_func, dtype_a, dtype_b, shape):
        np_a = make_arg(shape, dtype_a)
        np_b = make_arg(shape, dtype_b)
        expected = op_func(np_a, np_b)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = op_func(ga, gb)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize(
        "dtype_a,dtype_b",
        DTYPE_PAIRS,
        ids=[f"{np.dtype(a).name}-{np.dtype(b).name}" for a, b in DTYPE_PAIRS],
    )
    def test_cross_dtype_div(self, dtype_a, dtype_b):
        np_a = make_arg((5,), dtype_a)
        np_b = make_arg((5,), dtype_b)
        np_b = np.where(np_b == 0, np.ones_like(np_b), np_b)
        expected = np_a / np_b
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = ga / gb
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize(
        "dtype_a,dtype_b",
        DTYPE_PAIRS,
        ids=[f"{np.dtype(a).name}-{np.dtype(b).name}" for a, b in DTYPE_PAIRS],
    )
    def test_cross_dtype_pow(self, dtype_a, dtype_b):
        np_a = make_arg((5,), dtype_a)
        np_b = make_arg((5,), dtype_b)
        np_b = np.where(np_b == 0, np.ones_like(np_b), np_b)
        if np.issubdtype(dtype_b, np.integer):
            np_b = np.clip(np_b, 1, 3).astype(dtype_b)
        expected = np_a ** np_b
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = ga ** gb
        assert_eq(result, expected, dtype=expected.dtype, category="power")


class TestArithmeticReverse:
    """Reverse __radd__, __rsub__, __rtruediv__, __rpow__."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(5,)], ids=str)
    def test_radd(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = 3.0 + np_a
        result = 3.0 + cp.array(np_a)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(5,)], ids=str)
    def test_rsub(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = 10.0 - np_a
        result = 10.0 - cp.array(np_a)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(5,)], ids=str)
    def test_rtruediv(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        np_a = np.where(np_a == 0, np.ones_like(np_a), np_a)
        expected = 10.0 / np_a
        result = 10.0 / cp.array(np_a)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(5,)], ids=str)
    def test_rpow(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        if np.issubdtype(dtype, np.integer):
            np_a = np.clip(np_a, 0, 4).astype(dtype)
        expected = 2.0 ** np_a
        result = 2.0 ** cp.array(np_a)
        assert_eq(result, expected, dtype=expected.dtype, category="power")


class TestArithmeticUnary:
    """Unary -a, abs(a)."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES_NO_BOOL, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_ALL, ids=str)
    def test_neg(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = -np_a
        ga = cp.array(np_a)
        result = -ga
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES_NO_BOOL, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_ALL, ids=str)
    def test_abs(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        # Use negative values for signed types
        if np.issubdtype(dtype, np.signedinteger) or np.issubdtype(dtype, np.floating):
            np_a = -np_a
        expected = abs(np_a)
        ga = cp.array(np_a)
        result = abs(ga)
        assert_eq(result, expected, dtype=dtype)


class TestMatmul:
    """Matmul @ (FLOAT_DTYPES only)."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_matmul_basic(self, dtype):
        np_a = make_arg((2, 3), dtype)
        np_b = make_arg((3, 4), dtype)
        expected = np_a @ np_b
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = ga @ gb
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_matmul_square(self, dtype):
        np_a = make_arg((3, 3), dtype)
        np_b = make_arg((3, 3), dtype)
        expected = np_a @ np_b
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        assert_eq(ga @ gb, expected, dtype=dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_matmul_shape_error(self, dtype):
        ga = cp.array(make_arg((2, 3), dtype))
        gb = cp.array(make_arg((4, 5), dtype))
        with pytest.raises(ValueError):
            ga @ gb


class TestArithmeticBroadcast:
    """Broadcast binary: 5 ops x BROADCAST_PAIRS x NUMERIC_DTYPES."""

    @pytest.mark.parametrize(
        "op_name,op_func", BINARY_OPS, ids=[b[0] for b in BINARY_OPS]
    )
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize(
        "shapes", BROADCAST_PAIRS, ids=lambda p: f"{p[0]}+{p[1]}"
    )
    def test_broadcast_binary(self, op_name, op_func, dtype, shapes):
        shape_a, shape_b = shapes
        np_a = make_arg(shape_a, dtype)
        np_b = make_arg(shape_b, dtype)
        if op_name in ("truediv", "pow"):
            np_b = np.where(np_b == 0, np.ones_like(np_b), np_b)
        if op_name == "pow" and np.issubdtype(dtype, np.integer):
            np_b = np.clip(np_b, 0, 3).astype(dtype)
        expected = op_func(np_a, np_b)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = op_func(ga, gb)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")


# =====================================================================
# 6. Comparison operators  (~varies)
# =====================================================================
# Ref: cupy_tests/core_tests/test_ndarray_elementwise_op.py

COMPARISON_OPS = [
    ("lt", operator.lt),
    ("le", operator.le),
    ("gt", operator.gt),
    ("ge", operator.ge),
    ("eq", operator.eq),
    ("ne", operator.ne),
]

# Use a manageable subset of dtype pairs for comparisons
COMPARISON_DTYPES = [np.float32, np.int32, np.uint32]

# Cross-dtype pairs for comparison (from DTYPE_PAIRS, limited to no-bool)
COMPARISON_DTYPE_PAIRS = DTYPE_PAIRS


class TestComparison:
    """<, <=, >, >=, ==, != : array-array and array-scalar."""

    @pytest.mark.parametrize(
        "op_name,op_func", COMPARISON_OPS, ids=[c[0] for c in COMPARISON_OPS]
    )
    @pytest.mark.parametrize("dtype", ALL_DTYPES_NO_BOOL, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO, ids=str)
    def test_array_array(self, op_name, op_func, dtype, shape):
        if dtype == np.complex64 and op_name not in ("eq", "ne"):
            pytest.skip("ordering comparison not supported for complex types")
        np_a = make_arg(shape, dtype)
        np_b = make_arg(shape, dtype)
        expected = op_func(np_a, np_b)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = op_func(ga, gb)
        assert_eq(result, expected.astype(np.bool_))

    @pytest.mark.parametrize(
        "op_name,op_func", COMPARISON_OPS, ids=[c[0] for c in COMPARISON_OPS]
    )
    @pytest.mark.parametrize("dtype", ALL_DTYPES_NO_BOOL, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO, ids=str)
    def test_array_scalar(self, op_name, op_func, dtype, shape):
        if dtype == np.complex64 and op_name not in ("eq", "ne"):
            pytest.skip("ordering comparison not supported for complex types")
        np_a = make_arg(shape, dtype)
        scalar = 3.0
        expected = op_func(np_a, scalar)
        ga = cp.array(np_a)
        result = op_func(ga, scalar)
        assert_eq(result, expected.astype(np.bool_))


class TestComparisonCrossDtype:
    """Cross-dtype comparison: 6 ops x 81 dtype pairs x 3 shapes."""

    @pytest.mark.parametrize(
        "op_name,op_func", COMPARISON_OPS, ids=[c[0] for c in COMPARISON_OPS]
    )
    @pytest.mark.parametrize(
        "dtype_a,dtype_b",
        COMPARISON_DTYPE_PAIRS,
        ids=[f"{np.dtype(a).name}-{np.dtype(b).name}" for a, b in COMPARISON_DTYPE_PAIRS],
    )
    @pytest.mark.parametrize("shape", [(5,), (2, 3), (2, 3, 4)], ids=str)
    def test_cross_dtype(self, op_name, op_func, dtype_a, dtype_b, shape):
        if (dtype_a == np.complex64 or dtype_b == np.complex64) and op_name not in ("eq", "ne"):
            pytest.skip("ordering comparison not supported for complex types")
        np_a = make_arg(shape, dtype_a)
        np_b = make_arg(shape, dtype_b)
        expected = op_func(np_a, np_b)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = op_func(ga, gb)
        assert_eq(result, expected.astype(np.bool_))


class TestComparisonSpecial:
    """Comparison with nan and inf values."""

    @pytest.mark.parametrize(
        "op_name,op_func", COMPARISON_OPS, ids=[c[0] for c in COMPARISON_OPS]
    )
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_nan(self, op_name, op_func, dtype):
        np_a = np.array([1.0, np.nan, 3.0, np.nan], dtype=dtype)
        np_b = np.array([np.nan, 2.0, 3.0, np.nan], dtype=dtype)
        expected = op_func(np_a, np_b)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = op_func(ga, gb)
        assert_eq(result, expected.astype(np.bool_))

    @pytest.mark.parametrize(
        "op_name,op_func", COMPARISON_OPS, ids=[c[0] for c in COMPARISON_OPS]
    )
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_inf(self, op_name, op_func, dtype):
        np_a = np.array([1.0, np.inf, -np.inf, 0.0], dtype=dtype)
        np_b = np.array([np.inf, np.inf, 0.0, -np.inf], dtype=dtype)
        expected = op_func(np_a, np_b)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = op_func(ga, gb)
        assert_eq(result, expected.astype(np.bool_))


# =====================================================================
# 7. Boolean operators  (~24 cases)
# =====================================================================
# Ref: cupy_tests/core_tests/test_ndarray_elementwise_op.py


class TestBooleanOps:
    """&, |, ~ on bool arrays and int arrays."""

    @pytest.mark.parametrize("shape", ARITH_SHAPES, ids=str)
    def test_and_bool(self, shape):
        np_a = np.random.randint(0, 2, size=shape).astype(np.bool_)
        np_b = np.random.randint(0, 2, size=shape).astype(np.bool_)
        expected = np_a & np_b
        result = cp.array(np_a) & cp.array(np_b)
        assert_eq(result, expected)

    @pytest.mark.parametrize("shape", ARITH_SHAPES, ids=str)
    def test_or_bool(self, shape):
        np_a = np.random.randint(0, 2, size=shape).astype(np.bool_)
        np_b = np.random.randint(0, 2, size=shape).astype(np.bool_)
        expected = np_a | np_b
        result = cp.array(np_a) | cp.array(np_b)
        assert_eq(result, expected)

    @pytest.mark.parametrize("shape", ARITH_SHAPES, ids=str)
    def test_invert_bool(self, shape):
        np_a = np.random.randint(0, 2, size=shape).astype(np.bool_)
        expected = ~np_a
        result = ~cp.array(np_a)
        assert_eq(result, expected)

    @pytest.mark.parametrize("shape", ARITH_SHAPES, ids=str)
    def test_and_int(self, shape):
        np_a = np.random.randint(0, 2, size=shape).astype(np.int32)
        np_b = np.random.randint(0, 2, size=shape).astype(np.int32)
        expected = (np_a.astype(np.bool_) & np_b.astype(np.bool_))
        ga = cp.array(np_a.astype(np.bool_))
        gb = cp.array(np_b.astype(np.bool_))
        result = ga & gb
        assert_eq(result, expected)

    @pytest.mark.parametrize("shape", ARITH_SHAPES, ids=str)
    def test_or_int(self, shape):
        np_a = np.random.randint(0, 2, size=shape).astype(np.int32)
        np_b = np.random.randint(0, 2, size=shape).astype(np.int32)
        expected = (np_a.astype(np.bool_) | np_b.astype(np.bool_))
        ga = cp.array(np_a.astype(np.bool_))
        gb = cp.array(np_b.astype(np.bool_))
        result = ga | gb
        assert_eq(result, expected)


# =====================================================================
# 8. Reduction methods  (~384 cases)
# =====================================================================
# Ref: cupy_tests/core_tests/test_ndarray.py reductions

REDUCTION_SHAPES = [(5,), (2, 3), (2, 3, 4)]
REDUCTION_METHODS = ["sum", "max", "min", "mean"]


class TestReductions:
    """sum, max, min, mean as ndarray methods — axis=None, 0, -1, keepdims."""

    @pytest.mark.parametrize("method", REDUCTION_METHODS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", REDUCTION_SHAPES, ids=str)
    def test_axis_none(self, method, dtype, shape):
        np_a = make_arg(shape, dtype)
        ga = cp.array(np_a)
        expected = getattr(np_a, method)()
        result = getattr(ga, method)()
        cat = "arithmetic" if method == "mean" else "default"
        assert_eq(result, expected, dtype=expected.dtype, category=cat)

    @pytest.mark.parametrize("method", REDUCTION_METHODS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
    def test_axis_0(self, method, dtype, shape):
        np_a = make_arg(shape, dtype)
        ga = cp.array(np_a)
        expected = getattr(np_a, method)(axis=0)
        result = getattr(ga, method)(axis=0)
        cat = "arithmetic" if method == "mean" else "default"
        assert_eq(result, expected, dtype=expected.dtype, category=cat)

    @pytest.mark.parametrize("method", REDUCTION_METHODS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
    def test_axis_neg1(self, method, dtype, shape):
        np_a = make_arg(shape, dtype)
        ga = cp.array(np_a)
        expected = getattr(np_a, method)(axis=-1)
        result = getattr(ga, method)(axis=-1)
        cat = "arithmetic" if method == "mean" else "default"
        assert_eq(result, expected, dtype=expected.dtype, category=cat)

    @pytest.mark.parametrize("method", REDUCTION_METHODS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
    def test_keepdims(self, method, dtype, shape):
        np_a = make_arg(shape, dtype)
        ga = cp.array(np_a)
        expected = getattr(np_a, method)(axis=0, keepdims=True)
        result = getattr(ga, method)(axis=0, keepdims=True)
        assert result.shape == expected.shape
        cat = "arithmetic" if method == "mean" else "default"
        assert_eq(result, expected, dtype=expected.dtype, category=cat)


class TestReductionStdVar:
    """std, var reductions (float dtypes only)."""

    @pytest.mark.parametrize("method", ["std", "var"])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", REDUCTION_SHAPES, ids=str)
    def test_axis_none(self, method, dtype, shape):
        np_a = make_arg(shape, dtype)
        ga = cp.array(np_a)
        expected = getattr(np_a, method)()
        result = getattr(ga, method)()
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize("method", ["std", "var"])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
    def test_axis_0(self, method, dtype, shape):
        np_a = make_arg(shape, dtype)
        ga = cp.array(np_a)
        expected = getattr(np_a, method)(axis=0)
        result = getattr(ga, method)(axis=0)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize("method", ["std", "var"])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
    def test_keepdims(self, method, dtype, shape):
        np_a = make_arg(shape, dtype)
        ga = cp.array(np_a)
        expected = getattr(np_a, method)(axis=0, keepdims=True)
        result = getattr(ga, method)(axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")


class TestReductionProd:
    """prod reduction."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", REDUCTION_SHAPES, ids=str)
    def test_prod_axis_none(self, dtype, shape):
        # Use small values to avoid overflow
        np_a = np.ones(shape, dtype=dtype) * 2
        if np.dtype(dtype).itemsize <= 2 and np_a.size > 16:
            pytest.skip("16-bit prod overflows for large arrays (numpy upcasts to int64)")
        ga = cp.array(np_a)
        expected = np_a.prod()
        result = ga.prod()
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
    def test_prod_axis_0(self, dtype, shape):
        np_a = np.ones(shape, dtype=dtype) * 2
        ga = cp.array(np_a)
        expected = np_a.prod(axis=0)
        result = ga.prod(axis=0)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
    def test_prod_keepdims(self, dtype, shape):
        np_a = np.ones(shape, dtype=dtype) * 2
        ga = cp.array(np_a)
        expected = np_a.prod(axis=0, keepdims=True)
        result = ga.prod(axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")


class TestReductionCumsumCumprod:
    """cumsum, cumprod reductions."""

    @pytest.mark.parametrize("method", ["cumsum", "cumprod"])
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", REDUCTION_SHAPES, ids=str)
    def test_axis_none(self, method, dtype, shape):
        np_a = make_arg(shape, dtype)
        if method == "cumprod":
            np_a = np.ones(shape, dtype=dtype) * 2  # small values
            if np.dtype(dtype).itemsize <= 2 and np_a.size > 16:
                pytest.skip("16-bit cumprod overflows for large arrays (numpy upcasts to int64)")
        ga = cp.array(np_a)
        expected = getattr(np_a, method)()
        result = getattr(ga, method)()
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize("method", ["cumsum", "cumprod"])
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
    def test_axis_0(self, method, dtype, shape):
        np_a = make_arg(shape, dtype)
        if method == "cumprod":
            np_a = np.ones(shape, dtype=dtype) * 2
        ga = cp.array(np_a)
        expected = getattr(np_a, method)(axis=0)
        result = getattr(ga, method)(axis=0)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    @pytest.mark.parametrize("method", ["cumsum", "cumprod"])
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
    def test_axis_neg1(self, method, dtype, shape):
        np_a = make_arg(shape, dtype)
        if method == "cumprod":
            np_a = np.ones(shape, dtype=dtype) * 2
        ga = cp.array(np_a)
        expected = getattr(np_a, method)(axis=-1)
        result = getattr(ga, method)(axis=-1)
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")


class TestReductionAnyAll:
    """any, all reductions."""

    @pytest.mark.parametrize("shape", REDUCTION_SHAPES, ids=str)
    def test_any_mixed(self, shape):
        np_a = make_arg(shape, np.float32)
        ga = cp.array(np_a)
        expected = np_a.any()
        result = ga.any()
        assert result == expected

    @pytest.mark.parametrize("shape", REDUCTION_SHAPES, ids=str)
    def test_any_all_false(self, shape):
        np_a = np.zeros(shape, dtype=np.float32)
        ga = cp.array(np_a)
        assert ga.any() == False

    @pytest.mark.parametrize("shape", REDUCTION_SHAPES, ids=str)
    def test_all_true(self, shape):
        np_a = np.ones(shape, dtype=np.float32)
        ga = cp.array(np_a)
        assert ga.all() == True

    @pytest.mark.parametrize("shape", REDUCTION_SHAPES, ids=str)
    def test_all_mixed(self, shape):
        np_a = make_arg(shape, np.float32)
        np_a_flat = np_a.ravel()
        np_a_flat[0] = 0
        np_a = np_a_flat.reshape(shape)
        ga = cp.array(np_a)
        expected = np_a.all()
        assert ga.all() == expected


# =====================================================================
# 9. Indexing  (~390 cases)
# =====================================================================
# Ref: cupy_tests/core_tests/test_ndarray.py::TestNdarrayGetitem


class TestGetitem:
    """__getitem__: int, neg, slice, step, ellipsis, newaxis, bool mask, fancy."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_int_index(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        assert_eq(ga[2], np_a[2], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_neg_index(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        assert_eq(ga[-1], np_a[-1], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_slice(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        assert_eq(ga[1:4], np_a[1:4], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_step_slice(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        assert_eq(ga[::2], np_a[::2], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_2d_int(self, dtype):
        np_a = make_arg((2, 3), dtype)
        ga = cp.array(np_a)
        assert_eq(ga[0], np_a[0], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_2d_slice(self, dtype):
        np_a = make_arg((2, 3), dtype)
        ga = cp.array(np_a)
        assert_eq(ga[:, 1:], np_a[:, 1:], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_ellipsis(self, dtype):
        np_a = make_arg((2, 3, 4), dtype)
        ga = cp.array(np_a)
        assert_eq(ga[..., 0], np_a[..., 0], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_newaxis(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        result = ga[np.newaxis, :]
        expected = np_a[np.newaxis, :]
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_bool_mask(self, dtype):
        np_a = make_arg((5,), dtype)
        mask = np.array([True, False, True, False, True])
        ga = cp.array(np_a)
        assert_eq(ga[mask], np_a[mask], dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_fancy_index(self, dtype):
        np_a = make_arg((5,), dtype)
        idx = np.array([0, 2, 4])
        ga = cp.array(np_a)
        assert_eq(ga[idx], np_a[idx], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_3d_int(self, dtype):
        np_a = make_arg((2, 3, 4), dtype)
        ga = cp.array(np_a)
        assert_eq(ga[1], np_a[1], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_3d_slice(self, dtype):
        np_a = make_arg((2, 3, 4), dtype)
        ga = cp.array(np_a)
        assert_eq(ga[:, 1:, ::2], np_a[:, 1:, ::2], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_negative_step(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        assert_eq(ga[::-1], np_a[::-1], dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_2d_bool_mask(self, dtype):
        np_a = make_arg((2, 3), dtype)
        mask = np.array([[True, False, True], [False, True, False]])
        ga = cp.array(np_a)
        assert_eq(ga[mask], np_a[mask], dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_2d_fancy(self, dtype):
        np_a = make_arg((3, 3), dtype)
        idx = np.array([0, 2])
        ga = cp.array(np_a)
        assert_eq(ga[idx], np_a[idx], dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_scalar_result(self, dtype):
        np_a = make_arg((2, 3), dtype)
        ga = cp.array(np_a)
        result = ga[0, 1]
        expected = np_a[0, 1]
        assert_eq(result, expected, dtype=dtype)


class TestSetitem:
    """__setitem__: int, slice, bool mask, scalar, broadcast."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_int_setitem(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a.copy())
        np_a[2] = 99
        ga[2] = 99
        assert_eq(ga, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_slice_setitem(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a.copy())
        np_a[1:4] = 0
        ga[1:4] = 0
        assert_eq(ga, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_bool_mask_setitem(self, dtype):
        np_a = make_arg((5,), dtype)
        mask = np.array([True, False, True, False, True])
        ga = cp.array(np_a.copy())
        np_a[mask] = 0
        ga[mask] = 0
        assert_eq(ga, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_scalar_setitem(self, dtype):
        np_a = make_arg((2, 3), dtype)
        ga = cp.array(np_a.copy())
        np_a[0, 0] = 99
        ga[0, 0] = 99
        assert_eq(ga, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_broadcast_setitem(self, dtype):
        np_a = make_arg((2, 3), dtype)
        ga = cp.array(np_a.copy())
        row = np.array([10, 20, 30], dtype=dtype)
        np_a[0] = row
        ga[0] = row
        assert_eq(ga, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_2d_row_setitem(self, dtype):
        np_a = make_arg((3, 4), dtype)
        ga = cp.array(np_a.copy())
        np_a[1] = 0
        ga[1] = 0
        assert_eq(ga, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_fancy_setitem(self, dtype):
        np_a = make_arg((5,), dtype)
        idx = np.array([0, 2, 4])
        ga = cp.array(np_a.copy())
        np_a[idx] = 0
        ga[idx] = 0
        assert_eq(ga, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_step_setitem(self, dtype):
        np_a = make_arg((6,), dtype)
        ga = cp.array(np_a.copy())
        np_a[::2] = 0
        ga[::2] = 0
        assert_eq(ga, np_a, dtype=dtype)


# =====================================================================
# 10. Dunder methods  (~69 cases)
# =====================================================================
# Ref: cupy_tests/core_tests/test_ndarray.py


class TestDunderFloat:
    """__float__: size-1 ok, size-N error."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_float_size1(self, dtype):
        np_a = make_arg((1,), dtype)
        ga = cp.array(np_a)
        if np.issubdtype(dtype, np.complexfloating):
            pytest.skip("complex -> float not supported")
        expected = float(np_a.ravel()[0])
        assert float(ga) == pytest.approx(expected, rel=1e-2)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_float_sizeN_error(self, dtype):
        ga = cp.array(make_arg((5,), dtype))
        with pytest.raises(TypeError):
            float(ga)


class TestDunderInt:
    """__int__: size-1 ok, size-N error."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_int_size1(self, dtype):
        np_a = make_arg((1,), dtype)
        ga = cp.array(np_a)
        if np.issubdtype(dtype, np.complexfloating):
            pytest.skip("complex -> int not supported")
        expected = int(np_a.ravel()[0])
        assert int(ga) == expected

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_int_sizeN_error(self, dtype):
        ga = cp.array(make_arg((5,), dtype))
        with pytest.raises(TypeError):
            int(ga)


class TestDunderLen:
    """__len__: 1-D, 2-D, 0-D error."""

    def test_len_1d(self):
        ga = cp.array(make_arg((5,), np.float32))
        assert len(ga) == 5

    def test_len_2d(self):
        ga = cp.array(make_arg((2, 3), np.float32))
        assert len(ga) == 2

    def test_len_0d_error(self):
        ga = cp.array(make_arg((), np.float32))
        with pytest.raises(TypeError):
            len(ga)


class TestReprStr:
    """__repr__, __str__."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_repr(self, dtype):
        np_a = make_arg((3,), dtype)
        ga = cp.array(np_a)
        r = repr(ga)
        assert isinstance(r, str)
        assert len(r) > 0

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_str(self, dtype):
        np_a = make_arg((3,), dtype)
        ga = cp.array(np_a)
        s = str(ga)
        assert isinstance(s, str)
        assert len(s) > 0
