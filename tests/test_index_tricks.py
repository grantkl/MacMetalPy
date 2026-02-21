"""Tests for index expression objects -- c_, r_, s_, mgrid, ogrid.

Ref: numpy.c_, numpy.r_, numpy.s_, numpy.mgrid, numpy.ogrid
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

import macmetalpy as cp
from macmetalpy.ndarray import ndarray
from macmetalpy.index_tricks import c_, r_, s_, mgrid, ogrid


# ====================================================================
# c_ (CClass) -- column-wise concatenation
# ====================================================================

class TestC:
    def test_basic_arrays(self):
        result = c_[cp.array([1, 2, 3]), cp.array([4, 5, 6])]
        expected = np.c_[np.array([1, 2, 3]), np.array([4, 5, 6])]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_slice_range(self):
        result = c_[1:5]
        expected = np.c_[1:5]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_same_length_arrays(self):
        result = c_[cp.array([1, 2, 3]), cp.array([7, 8, 9])]
        expected = np.c_[np.array([1, 2, 3]), np.array([7, 8, 9])]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_single_element_arrays(self):
        result = c_[cp.array([1]), cp.array([2])]
        expected = np.c_[np.array([1]), np.array([2])]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_2d_arrays(self):
        a = cp.array(np.array([[1, 2], [3, 4]], dtype=np.int64))
        b = cp.array(np.array([[5, 6], [7, 8]], dtype=np.int64))
        result = c_[a, b]
        expected = np.c_[
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([[5, 6], [7, 8]], dtype=np.int64),
        ]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)


# ====================================================================
# r_ (RClass) -- row-wise concatenation
# ====================================================================

class TestR:
    def test_basic_arrays(self):
        result = r_[cp.array([1, 2, 3]), cp.array([4, 5, 6])]
        expected = np.r_[np.array([1, 2, 3]), np.array([4, 5, 6])]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_slice_range(self):
        result = r_[1:5]
        expected = np.r_[1:5]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_multiple_slices(self):
        result = r_[1:4, 0, 5:8]
        expected = np.r_[1:4, 0, 5:8]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_with_step(self):
        result = r_[0:10:2]
        expected = np.r_[0:10:2]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_scalar(self):
        result = r_[cp.array([1, 2, 3]), 4]
        expected = np.r_[np.array([1, 2, 3]), 4]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_string_directive(self):
        # r_ supports string directives like '0,2' to set axis and min dims
        result = r_['0,2', [1, 2, 3], [4, 5, 6]]
        expected = np.r_['0,2', [1, 2, 3], [4, 5, 6]]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)


# ====================================================================
# s_ (IndexExpression) -- index expression helper
# ====================================================================

class TestS:
    def test_single_slice(self):
        result = s_[1:5]
        expected = np.s_[1:5]
        assert result == expected

    def test_multi_dim_slice(self):
        result = s_[1:5, 2:8]
        expected = np.s_[1:5, 2:8]
        assert result == expected

    def test_integer_index(self):
        result = s_[3]
        expected = np.s_[3]
        assert result == expected

    def test_ellipsis(self):
        result = s_[..., 1:5]
        expected = np.s_[..., 1:5]
        assert result == expected

    def test_step(self):
        result = s_[1:10:2]
        expected = np.s_[1:10:2]
        assert result == expected

    def test_newaxis(self):
        result = s_[:, np.newaxis]
        expected = np.s_[:, np.newaxis]
        assert result == expected


# ====================================================================
# mgrid -- mesh grid via __getitem__
# ====================================================================

class TestMgrid:
    def test_basic_2d(self):
        result = mgrid[0:3, 0:4]
        expected = np.mgrid[0:3, 0:4]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert isinstance(r, ndarray)
            assert_array_equal(r.get(), e)

    def test_1d(self):
        result = mgrid[0:5]
        expected = np.mgrid[0:5]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_with_step(self):
        result = mgrid[0:5:2, 0:4:1]
        expected = np.mgrid[0:5:2, 0:4:1]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert isinstance(r, ndarray)
            assert_array_equal(r.get(), e)

    def test_float_step(self):
        result = mgrid[0:1:0.5]
        expected = np.mgrid[0:1:0.5]
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_complex_step(self):
        # complex step like 5j means 5 points between start and stop
        result = mgrid[0:1:5j]
        expected = np.mgrid[0:1:5j]
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)


# ====================================================================
# ogrid -- open mesh grid via __getitem__
# ====================================================================

class TestOgrid:
    def test_basic_2d(self):
        result = ogrid[0:3, 0:4]
        expected = np.ogrid[0:3, 0:4]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert isinstance(r, ndarray)
            assert_array_equal(r.get(), e)

    def test_1d(self):
        result = ogrid[0:5]
        expected = np.ogrid[0:5]
        assert isinstance(result, ndarray)
        assert_array_equal(result.get(), expected)

    def test_with_step(self):
        result = ogrid[0:5:2, 0:4:1]
        expected = np.ogrid[0:5:2, 0:4:1]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert isinstance(r, ndarray)
            assert_array_equal(r.get(), e)

    def test_shape_2d(self):
        result = ogrid[0:3, 0:4]
        # ogrid returns open meshgrid, so shapes should be (3,1) and (1,4)
        assert result[0].shape == (3, 1)
        assert result[1].shape == (1, 4)

    def test_complex_step(self):
        result = ogrid[0:1:5j]
        expected = np.ogrid[0:1:5j]
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)
