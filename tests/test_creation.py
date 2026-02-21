"""Tests for array creation functions.

Ref: cupy_tests/creation_tests/test_basic.py
Target: ~1,866 parametrized cases.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import macmetalpy as cp
from conftest import (
    ALL_DTYPES,
    FLOAT_DTYPES,
    NUMERIC_DTYPES,
    SHAPES_ALL,
    SHAPES_NONZERO,
    assert_eq,
    assert_shape_dtype,
    make_arg,
)

# ── Helpers ──────────────────────────────────────────────────────

SHAPES_3 = [(5,), (2, 3), (2, 3, 4)]


# ── 1. Core creation: zeros, ones, empty, full ──────────────────
# Ref: cupy_tests/creation_tests/test_basic.py::TestZeros, TestOnes, TestEmpty, TestFull


class TestZeros:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_values(self, dtype, shape):
        result = cp.zeros(shape, dtype=dtype)
        expected = np.zeros(shape, dtype=dtype)
        assert result.shape == shape
        assert result.dtype == np.dtype(dtype)
        assert_eq(result, expected, dtype=dtype)

    def test_default_dtype(self):
        result = cp.zeros((3,))
        assert result.dtype == np.float32

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_1d(self, dtype):
        result = cp.zeros((5,), dtype=dtype)
        expected = np.zeros((5,), dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_2d(self, dtype):
        result = cp.zeros((3, 4), dtype=dtype)
        expected = np.zeros((3, 4), dtype=dtype)
        assert result.shape == (3, 4)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_3d(self, dtype):
        result = cp.zeros((2, 3, 4), dtype=dtype)
        expected = np.zeros((2, 3, 4), dtype=dtype)
        assert result.shape == (2, 3, 4)
        assert_eq(result, expected, dtype=dtype)


class TestOnes:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_values(self, dtype, shape):
        result = cp.ones(shape, dtype=dtype)
        expected = np.ones(shape, dtype=dtype)
        assert result.shape == shape
        assert result.dtype == np.dtype(dtype)
        assert_eq(result, expected, dtype=dtype)

    def test_default_dtype(self):
        result = cp.ones((3,))
        assert result.dtype == np.float32

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_1d(self, dtype):
        result = cp.ones((5,), dtype=dtype)
        expected = np.ones((5,), dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_2d(self, dtype):
        result = cp.ones((3, 4), dtype=dtype)
        expected = np.ones((3, 4), dtype=dtype)
        assert result.shape == (3, 4)
        assert_eq(result, expected, dtype=dtype)


class TestEmpty:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_shape_dtype(self, dtype, shape):
        result = cp.empty(shape, dtype=dtype)
        assert result.shape == shape
        assert result.dtype == np.dtype(dtype)

    def test_default_dtype(self):
        result = cp.empty((3,))
        assert result.dtype == np.float32

    def test_does_not_error(self):
        result = cp.empty((2, 3), dtype=np.float32)
        _ = result.get()  # Should not raise


class TestFull:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_scalar(self, dtype, shape):
        fill = 7
        result = cp.full(shape, fill, dtype=dtype)
        expected = np.full(shape, fill, dtype=dtype)
        assert result.shape == shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_zero(self, dtype, shape):
        result = cp.full(shape, 0, dtype=dtype)
        expected = np.full(shape, 0, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_nan(self, dtype, shape):
        result = cp.full(shape, np.nan, dtype=dtype)
        expected = np.full(shape, np.nan, dtype=dtype)
        assert np.all(np.isnan(result.get()) == np.isnan(expected))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_inf(self, dtype, shape):
        result = cp.full(shape, np.inf, dtype=dtype)
        expected = np.full(shape, np.inf, dtype=dtype)
        assert np.all(np.isinf(result.get()) == np.isinf(expected))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_negative(self, dtype, shape):
        if np.issubdtype(dtype, np.unsignedinteger):
            return  # skip negative for unsigned
        fill = -5
        result = cp.full(shape, fill, dtype=dtype)
        expected = np.full(shape, fill, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    def test_default_dtype(self):
        result = cp.full((3,), 1.0)
        assert result.dtype == np.float32

    def test_float_fill(self):
        result = cp.full((3, 4), 3.14, dtype=np.float32)
        expected = np.full((3, 4), 3.14, dtype=np.float32)
        assert_eq(result, expected, dtype=np.float32)


# ── 2. Arange ───────────────────────────────────────────────────
# Ref: cupy_tests/creation_tests/test_ranges.py


class TestArange:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_int_step(self, dtype):
        result = cp.arange(0, 10, 2, dtype=dtype)
        expected = np.arange(0, 10, 2, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_float_step(self, dtype):
        result = cp.arange(0, 1, 0.25, dtype=dtype)
        expected = np.arange(0, 1, 0.25, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_negative_step(self, dtype):
        if np.issubdtype(dtype, np.unsignedinteger):
            return  # skip for unsigned
        result = cp.arange(10, 0, -2, dtype=dtype)
        expected = np.arange(10, 0, -2, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_large_range(self, dtype):
        result = cp.arange(0, 100, dtype=dtype)
        expected = np.arange(0, 100, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_start_equals_stop(self, dtype):
        result = cp.arange(5, 5, dtype=dtype)
        expected = np.arange(5, 5, dtype=dtype)
        assert result.shape == expected.shape

    def test_stop_only(self):
        result = cp.arange(10)
        assert result.shape[0] == 10

    def test_start_stop(self):
        result = cp.arange(2, 8)
        expected = np.arange(2, 8, dtype=np.float32)
        assert_eq(result, expected, dtype=np.float32)


# ── 3. Array / asarray ─────────────────────────────────────────
# Ref: cupy_tests/creation_tests/test_from_data.py


class TestArray:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_from_list(self, dtype):
        if dtype == np.bool_:
            data = [True, False, True]
        elif np.issubdtype(dtype, np.complexfloating):
            data = [1+2j, 3+4j, 5+6j]
        else:
            data = [1.0, 2.0, 3.0]
        result = cp.array(data, dtype=dtype)
        expected = np.array(data, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_from_nested_list(self, dtype):
        if dtype == np.bool_:
            data = [[True, False], [False, True]]
        elif np.issubdtype(dtype, np.complexfloating):
            data = [[1+0j, 2+0j], [3+0j, 4+0j]]
        else:
            data = [[1, 2], [3, 4]]
        result = cp.array(data, dtype=dtype)
        expected = np.array(data, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_from_tuple(self, dtype):
        if dtype == np.bool_:
            data = (True, False, True)
        elif np.issubdtype(dtype, np.complexfloating):
            data = (1+0j, 2+0j, 3+0j)
        else:
            data = (1.0, 2.0, 3.0)
        result = cp.array(data, dtype=dtype)
        expected = np.array(data, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_from_ndarray(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        result = cp.array(np_a)
        assert_eq(result, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_scalar(self, dtype):
        if dtype == np.bool_:
            val = True
        elif np.issubdtype(dtype, np.complexfloating):
            val = 1+2j
        else:
            val = 42
        result = cp.array(val, dtype=dtype)
        expected = np.array(val, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_empty(self, dtype):
        result = cp.array([], dtype=dtype)
        expected = np.array([], dtype=dtype)
        assert result.shape == expected.shape

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_dtype_override(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.array(np_a, dtype=dtype)
        expected = np.array(np_a, dtype=dtype)
        assert result.dtype == np.dtype(dtype)
        assert_eq(result, expected, dtype=dtype)

    def test_roundtrip(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = cp.array(np_a).get()
        assert_array_equal(result, np_a)

    def test_from_macmetalpy(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu_a = cp.array(np_a)
        gpu_b = cp.array(gpu_a)
        assert_eq(gpu_b, np_a, dtype=np.float32)


class TestAsarray:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_from_numpy(self, dtype):
        np_a = make_arg((5,), dtype)
        result = cp.asarray(np_a)
        assert_eq(result, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_preserves_dtype(self, dtype):
        np_a = make_arg((5,), dtype)
        result = cp.asarray(np_a)
        assert result.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_conversion(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        result = cp.asarray(np_a, dtype=dtype)
        assert_eq(result, np_a, dtype=dtype)

    def test_passthrough(self):
        gpu_a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.asarray(gpu_a)
        # Should be the same object (no copy)
        assert_eq(result, gpu_a.get(), dtype=np.float32)


# ── 4. *_like functions ─────────────────────────────────────────
# Ref: cupy_tests/creation_tests/test_basic.py


class TestZerosLike:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_same_dtype(self, dtype, shape):
        src = cp.ones(shape, dtype=dtype)
        result = cp.zeros_like(src)
        expected = np.zeros(shape, dtype=dtype)
        assert result.shape == shape
        assert result.dtype == np.dtype(dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_override_dtype(self, dtype, shape):
        src = cp.ones(shape, dtype=np.float32)
        result = cp.zeros_like(src, dtype=dtype)
        expected = np.zeros(shape, dtype=dtype)
        assert result.dtype == np.dtype(dtype)
        assert_eq(result, expected, dtype=dtype)


class TestOnesLike:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_same_dtype(self, dtype, shape):
        src = cp.zeros(shape, dtype=dtype)
        result = cp.ones_like(src)
        expected = np.ones(shape, dtype=dtype)
        assert result.shape == shape
        assert result.dtype == np.dtype(dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_override_dtype(self, dtype, shape):
        src = cp.zeros(shape, dtype=np.float32)
        result = cp.ones_like(src, dtype=dtype)
        expected = np.ones(shape, dtype=dtype)
        assert result.dtype == np.dtype(dtype)
        assert_eq(result, expected, dtype=dtype)


class TestEmptyLike:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_same_dtype(self, dtype, shape):
        src = cp.ones(shape, dtype=dtype)
        result = cp.empty_like(src)
        assert result.shape == shape
        assert result.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_override_dtype(self, dtype, shape):
        src = cp.ones(shape, dtype=np.float32)
        result = cp.empty_like(src, dtype=dtype)
        assert result.shape == shape
        assert result.dtype == np.dtype(dtype)


class TestFullLike:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_same_dtype(self, dtype, shape):
        src = cp.zeros(shape, dtype=dtype)
        fill = 5
        result = cp.full_like(src, fill)
        expected = np.full(shape, fill, dtype=dtype)
        assert result.shape == shape
        assert result.dtype == np.dtype(dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_override_dtype(self, dtype, shape):
        src = cp.zeros(shape, dtype=np.float32)
        fill = 5
        result = cp.full_like(src, fill, dtype=dtype)
        expected = np.full(shape, fill, dtype=dtype)
        assert result.dtype == np.dtype(dtype)
        assert_eq(result, expected, dtype=dtype)


# ── 5. Advanced creation ────────────────────────────────────────
# Ref: cupy_tests/creation_tests/test_ranges.py, test_matrix.py


class TestLinspace:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        result = cp.linspace(0, 10, num=50, dtype=dtype)
        expected = np.linspace(0, 10, num=50, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_num_1(self, dtype):
        result = cp.linspace(0, 10, num=1, dtype=dtype)
        expected = np.linspace(0, 10, num=1, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_num_0(self, dtype):
        result = cp.linspace(0, 10, num=0, dtype=dtype)
        expected = np.linspace(0, 10, num=0, dtype=dtype)
        assert result.shape == expected.shape

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative_range(self, dtype):
        result = cp.linspace(10, -10, num=21, dtype=dtype)
        expected = np.linspace(10, -10, num=21, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)


class TestEye:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_square(self, dtype):
        result = cp.eye(4, dtype=dtype)
        expected = np.eye(4, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_non_square(self, dtype):
        result = cp.eye(3, M=5, dtype=dtype)
        expected = np.eye(3, 5, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_n1(self, dtype):
        result = cp.eye(1, dtype=dtype)
        expected = np.eye(1, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_n4_default(self, dtype):
        result = cp.eye(4, dtype=dtype)
        expected = np.eye(4, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    def test_default_dtype(self):
        result = cp.eye(3)
        assert result.dtype == np.float32


class TestDiag:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d_to_2d(self, dtype):
        np_a = np.array([1, 2, 3], dtype=dtype)
        result = cp.diag(cp.array(np_a))
        expected = np.diag(np_a)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d_to_1d(self, dtype):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        result = cp.diag(cp.array(np_a))
        expected = np.diag(np_a)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_k_offset(self, dtype):
        np_a = np.array([1, 2, 3], dtype=dtype)
        result = cp.diag(cp.array(np_a), k=1)
        expected = np.diag(np_a, k=1)
        assert_eq(result, expected, dtype=dtype)


class TestIdentity:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_n1(self, dtype):
        result = cp.identity(1, dtype=dtype)
        expected = np.identity(1, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_n4(self, dtype):
        result = cp.identity(4, dtype=dtype)
        expected = np.identity(4, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)


class TestTri:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_square(self, dtype):
        result = cp.tri(4, dtype=dtype)
        expected = np.tri(4, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_non_square(self, dtype):
        result = cp.tri(3, M=5, dtype=dtype)
        expected = np.tri(3, 5, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_k_offset(self, dtype):
        result = cp.tri(4, k=1, dtype=dtype)
        expected = np.tri(4, k=1, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)


class TestTriu:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 1)])
    def test_k0(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        result = cp.triu(cp.array(np_a))
        expected = np.triu(np_a)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 1)])
    def test_k_positive(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        result = cp.triu(cp.array(np_a), k=1)
        expected = np.triu(np_a, k=1)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 1)])
    def test_k_negative(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        result = cp.triu(cp.array(np_a), k=-1)
        expected = np.triu(np_a, k=-1)
        assert_eq(result, expected, dtype=dtype)


class TestTril:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 1)])
    def test_k0(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        result = cp.tril(cp.array(np_a))
        expected = np.tril(np_a)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 1)])
    def test_k_positive(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        result = cp.tril(cp.array(np_a), k=1)
        expected = np.tril(np_a, k=1)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 1)])
    def test_k_negative(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        result = cp.tril(cp.array(np_a), k=-1)
        expected = np.tril(np_a, k=-1)
        assert_eq(result, expected, dtype=dtype)


class TestLogspace:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        result = cp.logspace(0, 2, num=10, dtype=dtype)
        expected = np.logspace(0, 2, num=10, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_num_small(self, dtype):
        result = cp.logspace(0, 1, num=3, dtype=dtype)
        expected = np.logspace(0, 1, num=3, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)


class TestMeshgrid:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2args(self, dtype):
        np_x = np.array([1, 2, 3], dtype=dtype)
        np_y = np.array([4, 5], dtype=dtype)
        exp_xx, exp_yy = np.meshgrid(np_x, np_y)
        res_xx, res_yy = cp.meshgrid(cp.array(np_x), cp.array(np_y))
        assert_eq(res_xx, exp_xx, dtype=dtype)
        assert_eq(res_yy, exp_yy, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_3args(self, dtype):
        np_x = np.array([1, 2], dtype=dtype)
        np_y = np.array([3, 4], dtype=dtype)
        np_z = np.array([5, 6], dtype=dtype)
        exp = np.meshgrid(np_x, np_y, np_z)
        res = cp.meshgrid(cp.array(np_x), cp.array(np_y), cp.array(np_z))
        for r, e in zip(res, exp):
            assert_eq(r, e, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_ij_indexing(self, dtype):
        np_x = np.array([1, 2, 3], dtype=dtype)
        np_y = np.array([4, 5], dtype=dtype)
        exp_xx, exp_yy = np.meshgrid(np_x, np_y, indexing='ij')
        res_xx, res_yy = cp.meshgrid(cp.array(np_x), cp.array(np_y), indexing='ij')
        assert_eq(res_xx, exp_xx, dtype=dtype)
        assert_eq(res_yy, exp_yy, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_xy_indexing(self, dtype):
        np_x = np.array([1, 2, 3], dtype=dtype)
        np_y = np.array([4, 5], dtype=dtype)
        exp_xx, exp_yy = np.meshgrid(np_x, np_y, indexing='xy')
        res_xx, res_yy = cp.meshgrid(cp.array(np_x), cp.array(np_y), indexing='xy')
        assert_eq(res_xx, exp_xx, dtype=dtype)
        assert_eq(res_yy, exp_yy, dtype=dtype)


class TestIndices:
    def test_2d(self):
        result = cp.indices((3, 4))
        expected = np.indices((3, 4))
        assert_eq(result, expected)

    def test_3d(self):
        result = cp.indices((2, 3, 4))
        expected = np.indices((2, 3, 4))
        assert_eq(result, expected)

    @pytest.mark.parametrize("dims", [(3, 4), (2, 3), (5, 2), (2, 3, 4)])
    def test_shapes(self, dims):
        result = cp.indices(dims)
        expected = np.indices(dims)
        assert_eq(result, expected)


class TestFromfunction:
    def test_1d(self):
        result = cp.fromfunction(lambda i: i * 2, (5,))
        expected = np.fromfunction(lambda i: i * 2, (5,))
        assert_eq(result, expected, dtype=np.float32)

    def test_2d(self):
        result = cp.fromfunction(lambda i, j: i + j, (3, 3))
        expected = np.fromfunction(lambda i, j: i + j, (3, 3))
        assert_eq(result, expected, dtype=np.float32)


class TestDiagflat:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_a = np.array([1, 2, 3], dtype=dtype)
        result = cp.diagflat(cp.array(np_a))
        expected = np.diagflat(np_a)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_k_offset(self, dtype):
        np_a = np.array([1, 2], dtype=dtype)
        result = cp.diagflat(cp.array(np_a), k=1)
        expected = np.diagflat(np_a, k=1)
        assert_eq(result, expected, dtype=dtype)


class TestVander:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        result = cp.vander(cp.array(np_a))
        expected = np.vander(np_a)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_n(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        result = cp.vander(cp.array(np_a), N=2)
        expected = np.vander(np_a, N=2)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_increasing(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        result = cp.vander(cp.array(np_a), increasing=True)
        expected = np.vander(np_a, increasing=True)
        assert_eq(result, expected, dtype=dtype)


class TestAsanyarray:
    def test_passthrough(self):
        gpu_a = cp.array([1.0, 2.0, 3.0])
        result = cp.asanyarray(gpu_a)
        assert_eq(result, gpu_a.get(), dtype=np.float32)

    def test_from_list(self):
        result = cp.asanyarray([1.0, 2.0, 3.0])
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert_eq(result, expected, dtype=np.float32)
