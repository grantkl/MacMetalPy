"""Tests for reduction parameter gap fixes (Task #1).

Tests all HIGH/MEDIUM severity param additions:
- out= for all/any/argmax/argmin/median/percentile/quantile/ptp/cumsum/cumprod
- keepdims= for amax/amin/argmax/argmin/median/percentile/quantile/ptp/average/count_nonzero
- dtype= for cumsum/cumprod
- initial= for sum/prod/max/min
- where= for sum/prod/mean/std/var/max/min
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp

from conftest import FLOAT_DTYPES, NUMERIC_DTYPES, assert_eq, tol_for


# ======================================================================
# 1. all() -- add out
# ======================================================================

class TestAllOut:
    def test_out_axis_none(self):
        a_np = np.ones((2, 3), dtype=np.float32)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.bool_)
        result = cp.all(a_cp, out=out)
        assert result is out
        assert bool(out.get()) == bool(np.all(a_np))

    def test_out_axis0(self):
        a_np = np.ones((2, 3), dtype=np.float32)
        a_np[0, 0] = 0
        a_cp = cp.array(a_np)
        expected = np.all(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.bool_)
        result = cp.all(a_cp, axis=0, out=out)
        assert result is out
        npt.assert_array_equal(out.get(), expected)


# ======================================================================
# 2. any() -- add out
# ======================================================================

class TestAnyOut:
    def test_out_axis_none(self):
        a_np = np.zeros((2, 3), dtype=np.float32)
        a_np[0, 0] = 1
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.bool_)
        result = cp.any(a_cp, out=out)
        assert result is out
        assert bool(out.get()) == bool(np.any(a_np))

    def test_out_axis0(self):
        a_np = np.zeros((2, 3), dtype=np.float32)
        a_np[0, 0] = 1
        a_cp = cp.array(a_np)
        expected = np.any(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.bool_)
        result = cp.any(a_cp, axis=0, out=out)
        assert result is out
        npt.assert_array_equal(out.get(), expected)


# ======================================================================
# 3. amax() -- add keepdims, out  (lives in ufunc_ops.py)
# ======================================================================

class TestAmaxKeepdims:
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims_axis0(self, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.amax(a_cp, axis=0, keepdims=True)
        expected = np.amax(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)

    def test_keepdims_axis_none(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.amax(a_cp, keepdims=True)
        expected = np.amax(a_np, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)


class TestAmaxOut:
    def test_out_axis_none(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.float32)
        result = cp.amax(a_cp, out=out)
        assert result is out
        npt.assert_allclose(out.get(), np.amax(a_np), **tol_for(np.float32))

    def test_out_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.amax(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.amax(a_cp, axis=0, out=out)
        assert result is out
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))


# ======================================================================
# 4. amin() -- add keepdims, out  (lives in ufunc_ops.py)
# ======================================================================

class TestAminKeepdims:
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims_axis0(self, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.amin(a_cp, axis=0, keepdims=True)
        expected = np.amin(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)

    def test_keepdims_axis_none(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.amin(a_cp, keepdims=True)
        expected = np.amin(a_np, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)


class TestAminOut:
    def test_out_axis_none(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.float32)
        result = cp.amin(a_cp, out=out)
        assert result is out
        npt.assert_allclose(out.get(), np.amin(a_np), **tol_for(np.float32))

    def test_out_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.amin(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.amin(a_cp, axis=0, out=out)
        assert result is out
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))


# ======================================================================
# 5. argmax() -- add keepdims, out
# ======================================================================

class TestArgmaxKeepdims:
    def test_keepdims_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        result = cp.argmax(a_cp, axis=0, keepdims=True)
        expected = np.argmax(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_array_equal(result.get(), expected)

    def test_keepdims_axis_none(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.argmax(a_cp, keepdims=True)
        expected = np.argmax(a_np, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_array_equal(result.get(), expected)


class TestArgmaxOut:
    def test_out_axis_none(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.int32)
        result = cp.argmax(a_cp, out=out)
        assert result is out
        assert int(out.get()) == int(np.argmax(a_np))

    def test_out_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.argmax(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.int32)
        result = cp.argmax(a_cp, axis=0, out=out)
        assert result is out
        npt.assert_array_equal(out.get(), expected)


# ======================================================================
# 6. argmin() -- add keepdims, out
# ======================================================================

class TestArgminKeepdims:
    def test_keepdims_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        result = cp.argmin(a_cp, axis=0, keepdims=True)
        expected = np.argmin(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_array_equal(result.get(), expected)

    def test_keepdims_axis_none(self):
        a_np = np.array([5, 1, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.argmin(a_cp, keepdims=True)
        expected = np.argmin(a_np, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_array_equal(result.get(), expected)


class TestArgminOut:
    def test_out_axis_none(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.int32)
        result = cp.argmin(a_cp, out=out)
        assert result is out
        assert int(out.get()) == int(np.argmin(a_np))

    def test_out_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.argmin(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.int32)
        result = cp.argmin(a_cp, axis=0, out=out)
        assert result is out
        npt.assert_array_equal(out.get(), expected)


# ======================================================================
# 7. median() -- add keepdims, out
# ======================================================================

class TestMedianKeepdims:
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims_axis0(self, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.median(a_cp, axis=0, keepdims=True)
        expected = np.median(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)

    def test_keepdims_axis_none(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.median(a_cp, keepdims=True)
        expected = np.median(a_np, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)


class TestMedianOut:
    def test_out_axis_none(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.float32)
        result = cp.median(a_cp, out=out)
        assert result is out
        npt.assert_allclose(out.get(), np.median(a_np), **tol_for(np.float32))

    def test_out_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.median(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.median(a_cp, axis=0, out=out)
        assert result is out
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))


# ======================================================================
# 8. percentile() -- add keepdims, out
# ======================================================================

class TestPercentileKeepdims:
    def test_keepdims_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        result = cp.percentile(a_cp, 50, axis=0, keepdims=True)
        expected = np.percentile(a_np, 50, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)

    def test_keepdims_axis_none(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.percentile(a_cp, 50, keepdims=True)
        expected = np.percentile(a_np, 50, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)


class TestPercentileOut:
    def test_out_axis_none(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.float32)
        result = cp.percentile(a_cp, 50, out=out)
        assert result is out
        npt.assert_allclose(out.get(), np.percentile(a_np, 50), **tol_for(np.float32))

    def test_out_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.percentile(a_np, 50, axis=0)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.percentile(a_cp, 50, axis=0, out=out)
        assert result is out
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))


# ======================================================================
# 9. quantile() -- add keepdims, out
# ======================================================================

class TestQuantileKeepdims:
    def test_keepdims_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        result = cp.quantile(a_cp, 0.5, axis=0, keepdims=True)
        expected = np.quantile(a_np, 0.5, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)

    def test_keepdims_axis_none(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.quantile(a_cp, 0.5, keepdims=True)
        expected = np.quantile(a_np, 0.5, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)


class TestQuantileOut:
    def test_out_axis_none(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.float32)
        result = cp.quantile(a_cp, 0.5, out=out)
        assert result is out
        npt.assert_allclose(out.get(), np.quantile(a_np, 0.5), **tol_for(np.float32))

    def test_out_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.quantile(a_np, 0.5, axis=0)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.quantile(a_cp, 0.5, axis=0, out=out)
        assert result is out
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))


# ======================================================================
# 10. ptp() -- add keepdims, out
# ======================================================================

class TestPtpKeepdims:
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims_axis0(self, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.ptp(a_cp, axis=0, keepdims=True)
        expected = np.ptp(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)

    def test_keepdims_axis_none(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.ptp(a_cp, keepdims=True)
        expected = np.ptp(a_np, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)


class TestPtpOut:
    def test_out_axis_none(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        out = cp.zeros((), dtype=np.float32)
        result = cp.ptp(a_cp, out=out)
        assert result is out
        npt.assert_allclose(out.get(), np.ptp(a_np), **tol_for(np.float32))

    def test_out_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.ptp(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.ptp(a_cp, axis=0, out=out)
        assert result is out
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))


# ======================================================================
# 11. average() -- add keepdims
# ======================================================================

class TestAverageKeepdims:
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims_axis0(self, shape):
        a_np = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
        a_cp = cp.array(a_np)
        result = cp.average(a_cp, axis=0, keepdims=True)
        expected = np.average(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)

    def test_keepdims_axis_none(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.average(a_cp, keepdims=True)
        expected = np.average(a_np, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)


# ======================================================================
# 12. count_nonzero() -- add keepdims
# ======================================================================

class TestCountNonzeroKeepdims:
    def test_keepdims_axis0(self):
        a_np = np.array([[0, 1, 0], [1, 1, 1]], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.count_nonzero(a_cp, axis=0, keepdims=True)
        expected = np.count_nonzero(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_array_equal(result.get(), expected)

    def test_keepdims_axis_none(self):
        a_np = np.array([[0, 1, 0], [1, 1, 1]], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.count_nonzero(a_cp, keepdims=True)
        expected = np.count_nonzero(a_np, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_array_equal(result.get(), expected)


# ======================================================================
# 13. cumsum() -- add dtype, out
# ======================================================================

class TestCumsumDtype:
    def test_int_to_float(self):
        a_np = np.array([1, 2, 3], dtype=np.int32)
        a_cp = cp.array(a_np)
        result = cp.cumsum(a_cp, dtype=np.float32)
        expected = np.cumsum(a_np, dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_float32_dtype(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.cumsum(a_cp, dtype=np.float32)
        expected = np.cumsum(a_np, dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


class TestCumsumOut:
    def test_out_axis_none(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        expected = np.cumsum(a_np)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.cumsum(a_cp, out=out)
        assert result is out
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))

    def test_out_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.cumsum(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.cumsum(a_cp, axis=0, out=out)
        assert result is out
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))


# ======================================================================
# 14. cumprod() -- add dtype, out
# ======================================================================

class TestCumprodDtype:
    def test_int_to_float(self):
        a_np = np.array([1, 2, 3], dtype=np.int32)
        a_cp = cp.array(a_np)
        result = cp.cumprod(a_cp, dtype=np.float32)
        expected = np.cumprod(a_np, dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_float32_dtype(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.cumprod(a_cp, dtype=np.float32)
        expected = np.cumprod(a_np, dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


class TestCumprodOut:
    def test_out_axis_none(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        expected = np.cumprod(a_np)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.cumprod(a_cp, out=out)
        assert result is out
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))

    def test_out_axis0(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        expected = np.cumprod(a_np, axis=0)
        out = cp.zeros(expected.shape, dtype=np.float32)
        result = cp.cumprod(a_cp, axis=0, out=out)
        assert result is out
        npt.assert_allclose(out.get(), expected, **tol_for(np.float32))


# ======================================================================
# 15. sum -- add initial, where
# ======================================================================

class TestSumInitial:
    def test_initial_scalar(self):
        a_np = np.array([1, 2, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.sum(a_cp, initial=10)
        expected = np.sum(a_np, initial=10)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_initial_with_axis(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        result = cp.sum(a_cp, axis=0, initial=5)
        expected = np.sum(a_np, axis=0, initial=5)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


class TestSumWhere:
    def test_where_mask(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        mask = np.array([True, False, True, False, True])
        a_cp = cp.array(a_np)
        mask_cp = cp.array(mask)
        result = cp.sum(a_cp, where=mask_cp)
        expected = np.sum(a_np, where=mask, initial=0)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_where_with_axis(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        mask = np.array([[True, False, True], [False, True, False]])
        a_cp = cp.array(a_np)
        mask_cp = cp.array(mask)
        result = cp.sum(a_cp, axis=0, where=mask_cp)
        expected = np.sum(a_np, axis=0, where=mask, initial=0)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# 16. prod -- add initial, where
# ======================================================================

class TestProdInitial:
    def test_initial_scalar(self):
        a_np = np.array([1, 2, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.prod(a_cp, initial=10)
        expected = np.prod(a_np, initial=10)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_initial_with_axis(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.prod(a_cp, axis=0, initial=2)
        expected = np.prod(a_np, axis=0, initial=2)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


class TestProdWhere:
    def test_where_mask(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        mask = np.array([True, False, True, False, True])
        a_cp = cp.array(a_np)
        mask_cp = cp.array(mask)
        result = cp.prod(a_cp, where=mask_cp)
        expected = np.prod(a_np, where=mask, initial=1)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# 17. mean -- add where
# ======================================================================

class TestMeanWhere:
    def test_where_mask(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        mask = np.array([True, False, True, False, True])
        a_cp = cp.array(a_np)
        mask_cp = cp.array(mask)
        result = cp.mean(a_cp, where=mask_cp)
        expected = np.mean(a_np, where=mask)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_where_with_axis(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        mask = np.array([[True, False, True], [True, True, False]])
        a_cp = cp.array(a_np)
        mask_cp = cp.array(mask)
        result = cp.mean(a_cp, axis=0, where=mask_cp)
        expected = np.mean(a_np, axis=0, where=mask)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# 18. std -- add where
# ======================================================================

class TestStdWhere:
    def test_where_mask(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        mask = np.array([True, False, True, False, True])
        a_cp = cp.array(a_np)
        mask_cp = cp.array(mask)
        result = cp.std(a_cp, where=mask_cp)
        expected = np.std(a_np, where=mask)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# 19. var -- add where
# ======================================================================

class TestVarWhere:
    def test_where_mask(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        mask = np.array([True, False, True, False, True])
        a_cp = cp.array(a_np)
        mask_cp = cp.array(mask)
        result = cp.var(a_cp, where=mask_cp)
        expected = np.var(a_np, where=mask)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# 20. max -- add initial, where
# ======================================================================

class TestMaxInitial:
    def test_initial_scalar(self):
        a_np = np.array([1, 2, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.max(a_cp, initial=10)
        expected = np.max(a_np, initial=10)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_initial_with_axis(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        result = cp.max(a_cp, axis=0, initial=10)
        expected = np.max(a_np, axis=0, initial=10)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


class TestMaxWhere:
    def test_where_mask(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        mask = np.array([True, False, True, False, True])
        a_cp = cp.array(a_np)
        mask_cp = cp.array(mask)
        result = cp.max(a_cp, where=mask_cp, initial=-np.inf)
        expected = np.max(a_np, where=mask, initial=-np.inf)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# 21. min -- add initial, where
# ======================================================================

class TestMinInitial:
    def test_initial_scalar(self):
        a_np = np.array([1, 2, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.min(a_cp, initial=-10)
        expected = np.min(a_np, initial=-10)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_initial_with_axis(self):
        a_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        result = cp.min(a_cp, axis=0, initial=-10)
        expected = np.min(a_np, axis=0, initial=-10)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


class TestMinWhere:
    def test_where_mask(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        mask = np.array([True, False, True, False, True])
        a_cp = cp.array(a_np)
        mask_cp = cp.array(mask)
        result = cp.min(a_cp, where=mask_cp, initial=np.inf)
        expected = np.min(a_np, where=mask, initial=np.inf)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# Backward compatibility: new params have defaults that preserve behavior
# ======================================================================

class TestBackwardCompatParams:
    """Calling functions without any new params still works."""

    def test_all_no_out(self):
        a = cp.array(np.ones(5, dtype=np.float32))
        result = cp.all(a)
        assert bool(result) is True

    def test_any_no_out(self):
        a = cp.array(np.zeros(5, dtype=np.float32))
        result = cp.any(a)
        assert bool(result) is False

    def test_amax_no_new_params(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.amax(a_cp), np.amax(a_np), dtype=np.float32)

    def test_amin_no_new_params(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.amin(a_cp), np.amin(a_np), dtype=np.float32)

    def test_argmax_no_new_params(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert int(cp.argmax(a_cp)) == int(np.argmax(a_np))

    def test_argmin_no_new_params(self):
        a_np = np.array([5, 1, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert int(cp.argmin(a_cp)) == int(np.argmin(a_np))

    def test_median_no_new_params(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.median(a_cp), np.median(a_np), dtype=np.float32)

    def test_percentile_no_new_params(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.percentile(a_cp, 50), np.percentile(a_np, 50), dtype=np.float32)

    def test_quantile_no_new_params(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.quantile(a_cp, 0.5), np.quantile(a_np, 0.5), dtype=np.float32)

    def test_ptp_no_new_params(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.ptp(a_cp), np.ptp(a_np), dtype=np.float32)

    def test_average_no_new_params(self):
        a_np = np.array([1, 5, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.average(a_cp), np.average(a_np), dtype=np.float32)

    def test_count_nonzero_no_keepdims(self):
        a_np = np.array([0, 1, 0, 1], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert cp.count_nonzero(a_cp) == np.count_nonzero(a_np)

    def test_cumsum_no_new_params(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.cumsum(a_cp), np.cumsum(a_np), dtype=np.float32)

    def test_cumprod_no_new_params(self):
        a_np = np.array([1, 2, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.cumprod(a_cp), np.cumprod(a_np), dtype=np.float32)

    def test_sum_no_initial_or_where(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.sum(a_cp), np.sum(a_np), dtype=np.float32)

    def test_prod_no_initial_or_where(self):
        a_np = np.array([1, 2, 3], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.prod(a_cp), np.prod(a_np), dtype=np.float32)

    def test_mean_no_where(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.mean(a_cp), np.mean(a_np), dtype=np.float32)

    def test_std_no_where(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.std(a_cp), np.std(a_np), dtype=np.float32)

    def test_var_no_where(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.var(a_cp), np.var(a_np), dtype=np.float32)

    def test_max_no_initial_or_where(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.max(a_cp), np.max(a_np), dtype=np.float32)

    def test_min_no_initial_or_where(self):
        a_np = np.arange(1, 6, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.min(a_cp), np.min(a_np), dtype=np.float32)
