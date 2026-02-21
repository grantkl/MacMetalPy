"""Tests for reduction parameter enhancements: ddof, out, dtype.

Covers:
- std/var with ddof parameter
- sum/mean/max/min/prod with out parameter
- sum/prod with dtype parameter
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp

from conftest import FLOAT_DTYPES, NUMERIC_DTYPES, assert_eq, tol_for


REDUCE_SHAPES = [(5,), (2, 3), (2, 3, 4)]


# ======================================================================
# std with ddof
# ======================================================================

class TestStdDdof:
    """Test std() ddof parameter -- sample vs population std."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_ddof0_matches_default(self, dtype, shape):
        """ddof=0 should match the default (population std)."""
        a_np = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
        a_cp = cp.array(a_np)
        result_default = cp.std(a_cp)
        result_ddof0 = cp.std(a_cp, ddof=0)
        npt.assert_allclose(result_default.get(), result_ddof0.get(), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_ddof1_axis_none(self, dtype, shape):
        """ddof=1 should match numpy sample std."""
        a_np = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.std(a_cp, ddof=1)
        expected = np.std(a_np, ddof=1)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_ddof1_axis0(self, dtype, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.std(a_cp, axis=0, ddof=1)
        expected = np.std(a_np, axis=0, ddof=1)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_ddof1_axis_neg1(self, dtype, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.std(a_cp, axis=-1, ddof=1)
        expected = np.std(a_np, axis=-1, ddof=1)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_ddof1_keepdims(self, dtype):
        a_np = np.arange(1, 7, dtype=dtype).reshape(2, 3)
        a_cp = cp.array(a_np)
        result = cp.std(a_cp, axis=0, keepdims=True, ddof=1)
        expected = np.std(a_np, axis=0, keepdims=True, ddof=1)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_ddof2(self, dtype):
        """ddof=2 for completeness."""
        a_np = np.arange(1, 11, dtype=dtype)
        a_cp = cp.array(a_np)
        result = cp.std(a_cp, ddof=2)
        expected = np.std(a_np, ddof=2)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


# ======================================================================
# var with ddof
# ======================================================================

class TestVarDdof:
    """Test var() ddof parameter -- sample vs population variance."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_ddof0_matches_default(self, dtype, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
        a_cp = cp.array(a_np)
        result_default = cp.var(a_cp)
        result_ddof0 = cp.var(a_cp, ddof=0)
        npt.assert_allclose(result_default.get(), result_ddof0.get(), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_ddof1_axis_none(self, dtype, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.var(a_cp, ddof=1)
        expected = np.var(a_np, ddof=1)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_ddof1_axis0(self, dtype, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.var(a_cp, axis=0, ddof=1)
        expected = np.var(a_np, axis=0, ddof=1)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_ddof1_axis_neg1(self, dtype, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.var(a_cp, axis=-1, ddof=1)
        expected = np.var(a_np, axis=-1, ddof=1)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_ddof1_keepdims(self, dtype):
        a_np = np.arange(1, 7, dtype=dtype).reshape(2, 3)
        a_cp = cp.array(a_np)
        result = cp.var(a_cp, axis=0, keepdims=True, ddof=1)
        expected = np.var(a_np, axis=0, keepdims=True, ddof=1)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_ddof2(self, dtype):
        a_np = np.arange(1, 11, dtype=dtype)
        a_cp = cp.array(a_np)
        result = cp.var(a_cp, ddof=2)
        expected = np.var(a_np, ddof=2)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


# ======================================================================
# out parameter for reductions
# ======================================================================

class TestSumOut:
    """Test sum() with out parameter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out_axis_none(self, dtype):
        a_np = np.arange(1, 6, dtype=dtype)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=dtype)
        result = cp.sum(a_cp, out=out)
        expected = np.sum(a_np)
        npt.assert_allclose(out.get(), expected, **tol_for(dtype))
        # result should be the same object as out
        assert result is out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out_axis0(self, dtype):
        a_np = np.arange(1, 7, dtype=dtype).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.sum(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=dtype)
        result = cp.sum(a_cp, axis=0, out=out)
        npt.assert_allclose(out.get(), expected, **tol_for(dtype))
        assert result is out


class TestMeanOut:
    """Test mean() with out parameter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out_axis_none(self, dtype):
        a_np = np.arange(1, 6, dtype=dtype)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.float32)
        result = cp.mean(a_cp, out=out)
        expected = np.mean(a_np)
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))
        assert result is out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out_axis0(self, dtype):
        a_np = np.arange(1, 7, dtype=dtype).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.mean(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.mean(a_cp, axis=0, out=out)
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))
        assert result is out


class TestMaxOut:
    """Test max() with out parameter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out_axis_none(self, dtype):
        a_np = np.arange(1, 6, dtype=dtype)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=dtype)
        result = cp.max(a_cp, out=out)
        expected = np.max(a_np)
        npt.assert_allclose(out.get(), expected, **tol_for(dtype))
        assert result is out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out_axis0(self, dtype):
        a_np = np.arange(1, 7, dtype=dtype).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.max(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=dtype)
        result = cp.max(a_cp, axis=0, out=out)
        npt.assert_allclose(out.get(), expected, **tol_for(dtype))
        assert result is out


class TestMinOut:
    """Test min() with out parameter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out_axis_none(self, dtype):
        a_np = np.arange(1, 6, dtype=dtype)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=dtype)
        result = cp.min(a_cp, out=out)
        expected = np.min(a_np)
        npt.assert_allclose(out.get(), expected, **tol_for(dtype))
        assert result is out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out_axis0(self, dtype):
        a_np = np.arange(1, 7, dtype=dtype).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.min(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=dtype)
        result = cp.min(a_cp, axis=0, out=out)
        npt.assert_allclose(out.get(), expected, **tol_for(dtype))
        assert result is out


class TestProdOut:
    """Test prod() with out parameter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out_axis_none(self, dtype):
        a_np = np.array([1.0, 2.0, 3.0], dtype=dtype)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=dtype)
        result = cp.prod(a_cp, out=out)
        expected = np.prod(a_np)
        npt.assert_allclose(out.get(), expected, **tol_for(dtype))
        assert result is out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out_axis0(self, dtype):
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        a_cp = cp.array(a_np)
        expected = np.prod(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=dtype)
        result = cp.prod(a_cp, axis=0, out=out)
        npt.assert_allclose(out.get(), expected, **tol_for(dtype))
        assert result is out


# ======================================================================
# dtype parameter for sum and prod
# ======================================================================

class TestSumDtype:
    """Test sum() with dtype parameter."""

    def test_int_to_float(self):
        a_np = np.array([1, 2, 3], dtype=np.int32)
        a_cp = cp.array(a_np)
        result = cp.sum(a_cp, dtype=np.float32)
        expected = np.sum(a_np, dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_float_dtype(self, dtype):
        a_np = np.arange(1, 6, dtype=dtype)
        a_cp = cp.array(a_np)
        result = cp.sum(a_cp, dtype=np.float32)
        expected = np.sum(a_np, dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype_with_axis(self, dtype):
        a_np = np.arange(1, 7, dtype=dtype).reshape(2, 3)
        a_cp = cp.array(a_np)
        result = cp.sum(a_cp, axis=0, dtype=np.float32)
        expected = np.sum(a_np, axis=0, dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


class TestProdDtype:
    """Test prod() with dtype parameter."""

    def test_int_to_float(self):
        a_np = np.array([1, 2, 3], dtype=np.int32)
        a_cp = cp.array(a_np)
        result = cp.prod(a_cp, dtype=np.float32)
        expected = np.prod(a_np, dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_float_dtype(self, dtype):
        a_np = np.array([1.0, 2.0, 3.0], dtype=dtype)
        a_cp = cp.array(a_np)
        result = cp.prod(a_cp, dtype=np.float32)
        expected = np.prod(a_np, dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# Backward compatibility: existing signatures still work
# ======================================================================

class TestBackwardCompat:
    """Ensure all new params have defaults matching old behavior."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_std_no_ddof(self, dtype):
        """Calling std without ddof should still work."""
        a_np = np.arange(1, 6, dtype=dtype)
        a_cp = cp.array(a_np)
        result = cp.std(a_cp)
        expected = np.std(a_np)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_var_no_ddof(self, dtype):
        a_np = np.arange(1, 6, dtype=dtype)
        a_cp = cp.array(a_np)
        result = cp.var(a_cp)
        expected = np.var(a_np)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sum_no_out(self, dtype):
        a_np = np.arange(1, 6, dtype=dtype)
        a_cp = cp.array(a_np)
        result = cp.sum(a_cp)
        expected = np.sum(a_np)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_no_out(self, dtype):
        a_np = np.arange(1, 6, dtype=dtype)
        a_cp = cp.array(a_np)
        result = cp.mean(a_cp)
        expected = np.mean(a_np)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))
