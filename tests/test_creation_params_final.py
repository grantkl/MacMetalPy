"""Tests for creation/math_ext/io gap-filling: new functions and param fixes."""

import tempfile
import os

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy import creation, math_ext, io
from macmetalpy.ndarray import ndarray
from conftest import FLOAT_DTYPES, NUMERIC_DTYPES, assert_eq


# ====================================================================
# creation.asarray_chkfinite
# ====================================================================

class TestAsarrayChkfinite:
    def test_basic_list(self):
        result = creation.asarray_chkfinite([1.0, 2.0, 3.0])
        expected = np.asarray_chkfinite([1.0, 2.0, 3.0], dtype=np.float32)
        assert isinstance(result, ndarray)
        assert_eq(result, expected, dtype=np.float32)

    def test_with_dtype(self):
        result = creation.asarray_chkfinite([1, 2, 3], dtype=np.int32)
        expected = np.asarray_chkfinite([1, 2, 3], dtype=np.int32)
        assert isinstance(result, ndarray)
        assert_eq(result, expected, dtype=np.int32)

    def test_raises_on_nan(self):
        with pytest.raises(ValueError):
            creation.asarray_chkfinite([1.0, float('nan'), 3.0])

    def test_raises_on_inf(self):
        with pytest.raises(ValueError):
            creation.asarray_chkfinite([1.0, float('inf'), 3.0])

    def test_raises_on_neg_inf(self):
        with pytest.raises(ValueError):
            creation.asarray_chkfinite([1.0, float('-inf'), 3.0])

    def test_from_ndarray(self):
        a = creation.array([1.0, 2.0, 3.0])
        result = creation.asarray_chkfinite(a)
        assert isinstance(result, ndarray)
        assert_eq(result, np.array([1.0, 2.0, 3.0], dtype=np.float32))


# ====================================================================
# creation.fromiter
# ====================================================================

class TestFromiter:
    def test_basic(self):
        result = creation.fromiter(iter([1.0, 2.0, 3.0]), dtype=np.float32)
        expected = np.fromiter(iter([1.0, 2.0, 3.0]), dtype=np.float32)
        assert isinstance(result, ndarray)
        assert_eq(result, expected, dtype=np.float32)

    def test_with_count(self):
        result = creation.fromiter(range(10), dtype=np.float32, count=5)
        expected = np.fromiter(range(10), dtype=np.float32, count=5)
        assert result.shape == (5,)
        assert_eq(result, expected, dtype=np.float32)

    def test_int_dtype(self):
        result = creation.fromiter(iter([1, 2, 3]), dtype=np.int32)
        expected = np.fromiter(iter([1, 2, 3]), dtype=np.int32)
        assert_eq(result, expected, dtype=np.int32)


# ====================================================================
# creation.fromstring
# ====================================================================

class TestFromstring:
    def test_basic_sep(self):
        result = creation.fromstring("1 2 3 4 5", dtype=np.float32, sep=" ")
        expected = np.fromstring("1 2 3 4 5", dtype=np.float32, sep=" ")
        assert isinstance(result, ndarray)
        assert_eq(result, expected, dtype=np.float32)

    def test_comma_sep(self):
        result = creation.fromstring("1,2,3", dtype=np.int32, sep=",")
        expected = np.fromstring("1,2,3", dtype=np.int32, sep=",")
        assert_eq(result, expected, dtype=np.int32)

    def test_with_count(self):
        result = creation.fromstring("1 2 3 4 5", dtype=np.float32, count=3, sep=" ")
        expected = np.fromstring("1 2 3 4 5", dtype=np.float32, count=3, sep=" ")
        assert result.shape == (3,)
        assert_eq(result, expected, dtype=np.float32)


# ====================================================================
# creation.indices — sparse= param
# ====================================================================

class TestIndicesSparse:
    def test_dense_default(self):
        result = creation.indices((2, 3))
        expected = np.indices((2, 3))
        assert isinstance(result, ndarray)
        # Dense returns a single array of shape (ndim, *dimensions)
        npt.assert_array_equal(result.get(), expected.astype(result.dtype))

    def test_sparse_true(self):
        result = creation.indices((2, 3), sparse=True)
        expected = np.indices((2, 3), sparse=True)
        assert isinstance(result, list)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert isinstance(r, ndarray)
            npt.assert_array_equal(r.get(), e.astype(r.dtype))

    def test_sparse_false(self):
        result = creation.indices((2, 3), sparse=False)
        expected = np.indices((2, 3), sparse=False)
        assert isinstance(result, ndarray)
        npt.assert_array_equal(result.get(), expected.astype(result.dtype))


# ====================================================================
# creation.asfarray
# ====================================================================

class TestAsfarray:
    def test_basic_list(self):
        result = creation.asfarray([1, 2, 3])
        assert isinstance(result, ndarray)
        # Default dtype is float64, which gets downcast to float32
        assert result.dtype == np.float32

    def test_from_int_array(self):
        result = creation.asfarray(np.array([1, 2, 3], dtype=np.int32))
        assert isinstance(result, ndarray)
        # Should be float
        assert np.issubdtype(result.dtype, np.floating)
        npt.assert_array_equal(result.get(), np.array([1.0, 2.0, 3.0], dtype=result.dtype))

    def test_with_float32_dtype(self):
        result = creation.asfarray([1, 2, 3], dtype=np.float32)
        assert isinstance(result, ndarray)
        assert result.dtype == np.float32
        npt.assert_array_equal(result.get(), np.array([1.0, 2.0, 3.0], dtype=np.float32))


# ====================================================================
# math_ext.interp — period= param
# ====================================================================

class TestInterpPeriod:
    def test_basic_no_period(self):
        x = creation.array([0.5, 1.5, 2.5])
        xp = creation.array([0.0, 1.0, 2.0, 3.0])
        fp = creation.array([0.0, 1.0, 2.0, 3.0])
        result = math_ext.interp(x, xp, fp)
        expected = np.interp([0.5, 1.5, 2.5], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])
        assert_eq(result, expected, dtype=np.float32)

    def test_with_period(self):
        x = creation.array([5.0, 7.0])
        xp = creation.array([0.0, 1.0, 2.0, 3.0])
        fp = creation.array([0.0, 10.0, 20.0, 30.0])
        result = math_ext.interp(x, xp, fp, period=4.0)
        expected = np.interp([5.0, 7.0], [0.0, 1.0, 2.0, 3.0],
                             [0.0, 10.0, 20.0, 30.0], period=4.0)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)


# ====================================================================
# math_ext.unwrap — period= param
# ====================================================================

class TestUnwrapPeriod:
    def test_basic_no_period(self):
        p = creation.array([0.0, 1.0, 2.0, 3.0 + 2 * np.pi, 4.0 + 2 * np.pi])
        result = math_ext.unwrap(p)
        expected = np.unwrap(p.get().astype(np.float64))
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)

    def test_with_period(self):
        p = creation.array([0.0, 1.0, 2.0, 3.0 + 10, 4.0 + 10])
        result = math_ext.unwrap(p, period=10)
        expected = np.unwrap(p.get().astype(np.float64), period=10)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)


# ====================================================================
# math_ext.piecewise
# ====================================================================

class TestPiecewise:
    def test_basic(self):
        x = creation.array([0.0, 1.0, 2.0, 3.0, 4.0])
        condlist = [x.get() < 2, x.get() >= 2]
        funclist = [lambda x: x, lambda x: x * 10]
        result = math_ext.piecewise(x, condlist, funclist)
        expected = np.piecewise(x.get(), condlist, funclist)
        assert isinstance(result, ndarray)
        npt.assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)

    def test_with_ndarray_conditions(self):
        x = creation.array([0.0, 1.0, 2.0, 3.0, 4.0])
        cond1 = creation.array([True, True, False, False, False])
        cond2 = creation.array([False, False, True, True, True])
        funclist = [0.0, 1.0]
        result = math_ext.piecewise(x, [cond1, cond2], funclist)
        expected = np.piecewise(x.get(), [cond1.get(), cond2.get()], funclist)
        assert isinstance(result, ndarray)
        npt.assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)


# ====================================================================
# io.load — encoding= and fix_imports= params
# ====================================================================

class TestLoadParams:
    def test_load_with_encoding(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, np.array([1.0, 2.0, 3.0], dtype=np.float32))
            result = io.load(f.name, encoding='ASCII')
            assert isinstance(result, ndarray)
            npt.assert_array_equal(result.get(), np.array([1.0, 2.0, 3.0], dtype=np.float32))
            os.unlink(f.name)

    def test_load_with_fix_imports(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, np.array([4.0, 5.0, 6.0], dtype=np.float32))
            result = io.load(f.name, fix_imports=True)
            assert isinstance(result, ndarray)
            npt.assert_array_equal(result.get(), np.array([4.0, 5.0, 6.0], dtype=np.float32))
            os.unlink(f.name)

    def test_load_with_all_params(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, np.array([7.0, 8.0, 9.0], dtype=np.float32))
            result = io.load(f.name, encoding='ASCII', fix_imports=True)
            assert isinstance(result, ndarray)
            npt.assert_array_equal(result.get(), np.array([7.0, 8.0, 9.0], dtype=np.float32))
            os.unlink(f.name)


# ====================================================================
# math_ext.spacing
# ====================================================================

class TestSpacing:
    def test_basic(self):
        x = creation.array([1.0])
        result = math_ext.spacing(x)
        # np.spacing on float32 input produces float32 spacing values
        expected = np.spacing(x.get())
        assert isinstance(result, ndarray)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_multiple_values(self):
        x = creation.array([0.0, 1.0, 100.0, 1000.0])
        result = math_ext.spacing(x)
        # np.spacing on float32 input produces float32 spacing values
        expected = np.spacing(x.get())
        assert isinstance(result, ndarray)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


# ====================================================================
# math_ext.isnat
# ====================================================================

class TestIsnat:
    def test_nat(self):
        x_np = np.array(['NaT', '2020-01-01'], dtype='datetime64')
        # isnat accepts raw numpy arrays since datetime64 is not a Metal dtype
        result = math_ext.isnat(x_np)
        expected = np.isnat(x_np)
        assert isinstance(result, ndarray)
        npt.assert_array_equal(result.get(), expected)
