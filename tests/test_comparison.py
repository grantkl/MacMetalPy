"""Tests for comparison operators and boolean logic."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import macmetalpy as cp


@pytest.fixture
def float_arrays():
    a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b_np = np.array([2.0, 2.0, 4.0, 3.0, 5.0], dtype=np.float32)
    return cp.array(a_np), cp.array(b_np), a_np, b_np


@pytest.fixture
def bool_arrays():
    a_np = np.array([True, False, True, False, True])
    b_np = np.array([True, True, False, False, True])
    return cp.array(a_np), cp.array(b_np), a_np, b_np


class TestComparisonFloat:
    def test_less_than(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a < b
        expected = a_np < b_np
        assert_array_equal(result.get(), expected)

    def test_less_equal(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a <= b
        expected = a_np <= b_np
        assert_array_equal(result.get(), expected)

    def test_greater_than(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a > b
        expected = a_np > b_np
        assert_array_equal(result.get(), expected)

    def test_greater_equal(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a >= b
        expected = a_np >= b_np
        assert_array_equal(result.get(), expected)

    def test_equal(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a == b
        expected = a_np == b_np
        assert_array_equal(result.get(), expected)

    def test_not_equal(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a != b
        expected = a_np != b_np
        assert_array_equal(result.get(), expected)


class TestComparisonScalar:
    def test_less_equal_scalar(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = a <= 3.0
        expected = a_np <= 3.0
        assert_array_equal(result.get(), expected)

    def test_greater_scalar(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = a > 3.0
        expected = a_np > 3.0
        assert_array_equal(result.get(), expected)

    def test_equal_scalar(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = a == 3.0
        expected = a_np == 3.0
        assert_array_equal(result.get(), expected)

    def test_not_equal_scalar(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = a != 3.0
        expected = a_np != 3.0
        assert_array_equal(result.get(), expected)


class TestBooleanLogic:
    def test_and(self, bool_arrays):
        a, b, a_np, b_np = bool_arrays
        result = a & b
        expected = a_np & b_np
        assert_array_equal(result.get(), expected)

    def test_or(self, bool_arrays):
        a, b, a_np, b_np = bool_arrays
        result = a | b
        expected = a_np | b_np
        assert_array_equal(result.get(), expected)

    def test_invert(self, bool_arrays):
        a, _, a_np, _ = bool_arrays
        result = ~a
        expected = ~a_np
        assert_array_equal(result.get(), expected)

    def test_ior(self, bool_arrays):
        a, b, a_np, b_np = bool_arrays
        a_np_copy = a_np.copy()
        a_copy = cp.array(a_np_copy)
        a_copy |= b
        a_np_copy |= b_np
        assert_array_equal(a_copy.get(), a_np_copy)


class TestMixedDtypeComparison:
    def test_float32_vs_int32(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_np = np.array([1, 3, 2], dtype=np.int32)
        a = cp.array(a_np)
        b = cp.array(b_np)
        result = a < b
        expected = a_np < b_np
        assert_array_equal(result.get(), expected)

    def test_int32_vs_float32(self):
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([1.5, 1.5, 1.5], dtype=np.float32)
        a = cp.array(a_np)
        b = cp.array(b_np)
        result = a >= b
        expected = a_np >= b_np
        assert_array_equal(result.get(), expected)


class TestTradingBotPatterns:
    def test_trading_bot_fill_mask(self):
        """Test the exact pattern used by backtest_fast.py."""
        pp = cp.array([10.0, 10.5, 10.3, 10.8, 10.1], dtype=cp.float32)
        pm = cp.array([1, 3, 2, 5, 1], dtype=cp.int32)
        limit = 10.4
        fill_window = 3

        pp_mask = pp <= limit
        pm_mask = pm <= fill_window

        fill_mask = pp_mask & pm_mask
        expected = np.array([True, False, True, False, True])
        assert_array_equal(fill_mask.get(), expected)

    def test_bool_slice_assignment(self):
        """Test boolean slice assignment: m[i:] = expr[i:]"""
        m = cp.zeros(5, dtype=cp.bool_)
        vals = cp.array([True, False, True, True, False], dtype=cp.bool_)
        m[2:] = vals[2:]
        expected = np.array([False, False, True, True, False])
        assert_array_equal(m.get(), expected)

    def test_ior_slice(self):
        """Test |= on slices (trading bot ratchet pattern)."""
        m = cp.array([True, False, False, False, False], dtype=cp.bool_)
        hits = cp.array([False, True, False], dtype=cp.bool_)
        np_m = m.get()
        np_m[2:] |= hits.get()
        m = cp.array(np_m)
        expected = np.array([True, False, False, True, False])
        assert_array_equal(m.get(), expected)

    def test_comparison_returns_bool_dtype(self):
        a = cp.array([1.0, 2.0], dtype=cp.float32)
        result = a < 5.0
        assert result.dtype == np.bool_
