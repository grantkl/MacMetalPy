"""Tests for the final remaining parameter gaps (HIGH + MEDIUM severity)."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

import macmetalpy as cp
from macmetalpy.ndarray import ndarray


# ================================================================== fix() out=

class TestFixOut:
    def test_fix_basic(self):
        a = cp.array([1.7, -2.3, 0.0, 3.9])
        result = cp.fix(a)
        expected = np.fix(np.array([1.7, -2.3, 0.0, 3.9]))
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_fix_out(self):
        a = cp.array([1.7, -2.3, 0.0, 3.9])
        out = cp.zeros(4, dtype=np.float32)
        result = cp.fix(a, out=out)
        assert result is out
        expected = np.fix(np.array([1.7, -2.3, 0.0, 3.9]))
        assert_allclose(out.get(), expected, rtol=1e-5)


# ================================================================== gradient() axis=

class TestGradientAxis:
    def test_gradient_1d(self):
        a = cp.array([1.0, 2.0, 4.0, 7.0, 11.0])
        result = cp.gradient(a)
        expected = np.gradient(np.array([1.0, 2.0, 4.0, 7.0, 11.0]))
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_gradient_2d_axis0(self):
        a_np = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]])
        a = cp.array(a_np)
        result = cp.gradient(a, axis=0)
        expected = np.gradient(a_np, axis=0)
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_gradient_2d_axis1(self):
        a_np = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 8.0]])
        a = cp.array(a_np)
        result = cp.gradient(a, axis=1)
        expected = np.gradient(a_np, axis=1)
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_gradient_edge_order(self):
        a_np = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
        a = cp.array(a_np)
        result = cp.gradient(a, edge_order=2)
        expected = np.gradient(a_np, edge_order=2)
        assert_allclose(result.get(), expected, rtol=1e-5)


# ================================================================== all() where=

class TestAllWhere:
    def test_all_basic(self):
        a = cp.array([True, True, True])
        result = cp.all(a)
        # scalar reduction may return bool directly
        assert bool(result) is True

    def test_all_where_mask(self):
        a = cp.array([True, False, True])
        mask = cp.array([True, False, True])  # ignore the False element
        result = cp.all(a, where=mask)
        expected = np.all(np.array([True, False, True]), where=np.array([True, False, True]))
        assert bool(result.get()) == bool(expected)

    def test_all_where_axis(self):
        a_np = np.array([[True, False], [True, True]])
        mask_np = np.array([[True, False], [True, True]])
        a = cp.array(a_np)
        mask = cp.array(mask_np)
        result = cp.all(a, axis=1, where=mask)
        expected = np.all(a_np, axis=1, where=mask_np)
        assert_array_equal(result.get(), expected)


# ================================================================== any() where=

class TestAnyWhere:
    def test_any_basic(self):
        a = cp.array([False, False, True])
        result = cp.any(a)
        assert bool(result) is True

    def test_any_where_mask(self):
        a = cp.array([False, True, False])
        mask = cp.array([True, False, True])  # ignore the True element
        result = cp.any(a, where=mask)
        expected = np.any(np.array([False, True, False]), where=np.array([True, False, True]))
        assert bool(result.get()) == bool(expected)

    def test_any_where_axis(self):
        a_np = np.array([[False, True], [False, False]])
        mask_np = np.array([[True, False], [True, True]])
        a = cp.array(a_np)
        mask = cp.array(mask_np)
        result = cp.any(a, axis=1, where=mask)
        expected = np.any(a_np, axis=1, where=mask_np)
        assert_array_equal(result.get(), expected)


# ================================================================== amax() initial/where

class TestAmaxInitialWhere:
    def test_amax_initial(self):
        a = cp.array([1.0, 2.0, 3.0])
        result = cp.amax(a, initial=5.0)
        assert float(result.get()) == 5.0

    def test_amax_initial_smaller(self):
        a = cp.array([1.0, 2.0, 3.0])
        result = cp.amax(a, initial=0.0)
        assert float(result.get()) == 3.0

    def test_amax_where(self):
        a = cp.array([1.0, 5.0, 3.0])
        mask = cp.array([True, False, True])  # ignore 5.0
        result = cp.amax(a, where=mask)
        # with where masking out index 1, max of [1.0, 3.0] = 3.0
        assert float(result.get()) == 3.0

    def test_amax_where_axis(self):
        a_np = np.array([[1.0, 5.0], [3.0, 2.0]])
        mask_np = np.array([[True, False], [True, True]])
        a = cp.array(a_np)
        mask = cp.array(mask_np)
        result = cp.amax(a, axis=1, where=mask, initial=-np.inf)
        expected = np.amax(a_np, axis=1, where=mask_np, initial=-np.inf)
        assert_allclose(result.get(), expected, rtol=1e-5)


# ================================================================== amin() initial/where

class TestAminInitialWhere:
    def test_amin_initial(self):
        a = cp.array([1.0, 2.0, 3.0])
        result = cp.amin(a, initial=-1.0)
        assert float(result.get()) == -1.0

    def test_amin_initial_larger(self):
        a = cp.array([1.0, 2.0, 3.0])
        result = cp.amin(a, initial=10.0)
        assert float(result.get()) == 1.0

    def test_amin_where(self):
        a = cp.array([5.0, 1.0, 3.0])
        mask = cp.array([True, False, True])  # ignore 1.0
        result = cp.amin(a, where=mask)
        assert float(result.get()) == 3.0  # min of [5.0, 3.0] with mask

    def test_amin_where_axis(self):
        a_np = np.array([[5.0, 1.0], [3.0, 2.0]])
        mask_np = np.array([[True, False], [True, True]])
        a = cp.array(a_np)
        mask = cp.array(mask_np)
        result = cp.amin(a, axis=1, where=mask, initial=np.inf)
        expected = np.amin(a_np, axis=1, where=mask_np, initial=np.inf)
        assert_allclose(result.get(), expected, rtol=1e-5)
