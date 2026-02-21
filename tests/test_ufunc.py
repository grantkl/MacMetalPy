"""Tests for ufunc-like objects (maximum, minimum)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp


class TestMaximumElementwise:
    def test_maximum_basic(self):
        a = cp.array([1.0, 5.0, 3.0], dtype=cp.float32)
        b = cp.array([2.0, 4.0, 6.0], dtype=cp.float32)
        result = cp.maximum(a, b)
        expected = np.maximum(
            np.array([1.0, 5.0, 3.0], dtype=np.float32),
            np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        assert_allclose(result.get(), expected)

    def test_maximum_int(self):
        a = cp.array([1, 5, 3], dtype=cp.int32)
        b = cp.array([2, 4, 6], dtype=cp.int32)
        result = cp.maximum(a, b)
        expected = np.array([2, 5, 6], dtype=np.int32)
        assert_array_equal(result.get(), expected)

    def test_maximum_same_values(self):
        a = cp.array([3.0, 3.0, 3.0], dtype=cp.float32)
        b = cp.array([3.0, 3.0, 3.0], dtype=cp.float32)
        result = cp.maximum(a, b)
        assert_allclose(result.get(), np.array([3.0, 3.0, 3.0], dtype=np.float32))


class TestMinimumElementwise:
    def test_minimum_basic(self):
        a = cp.array([1.0, 5.0, 3.0], dtype=cp.float32)
        b = cp.array([2.0, 4.0, 6.0], dtype=cp.float32)
        result = cp.minimum(a, b)
        expected = np.minimum(
            np.array([1.0, 5.0, 3.0], dtype=np.float32),
            np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        assert_allclose(result.get(), expected)

    def test_minimum_int(self):
        a = cp.array([1, 5, 3], dtype=cp.int32)
        b = cp.array([2, 4, 6], dtype=cp.int32)
        result = cp.minimum(a, b)
        expected = np.array([1, 4, 3], dtype=np.int32)
        assert_array_equal(result.get(), expected)


class TestMaximumAccumulate:
    def test_accumulate_basic(self):
        """Running maximum -- the trading bot's HWM pattern."""
        a = cp.array([10.0, 10.5, 10.3, 10.8, 10.1], dtype=cp.float32)
        result = cp.maximum.accumulate(a)
        expected = np.maximum.accumulate(
            np.array([10.0, 10.5, 10.3, 10.8, 10.1], dtype=np.float32)
        )
        assert_allclose(result.get(), expected)

    def test_accumulate_decreasing(self):
        a = cp.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=cp.float32)
        result = cp.maximum.accumulate(a)
        expected = np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        assert_allclose(result.get(), expected)

    def test_accumulate_increasing(self):
        a = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32)
        result = cp.maximum.accumulate(a)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        assert_allclose(result.get(), expected)

    def test_accumulate_int(self):
        a = cp.array([3, 1, 4, 1, 5], dtype=cp.int32)
        result = cp.maximum.accumulate(a)
        expected = np.array([3, 3, 4, 4, 5], dtype=np.int32)
        assert_array_equal(result.get(), expected)

    def test_accumulate_2d(self):
        a = cp.array([[1.0, 4.0], [3.0, 2.0]], dtype=cp.float32)
        result = cp.maximum.accumulate(a, axis=0)
        expected = np.array([[1.0, 4.0], [3.0, 4.0]], dtype=np.float32)
        assert_array_equal(result.get(), expected)


class TestMinimumAccumulate:
    def test_accumulate_basic(self):
        a = cp.array([5.0, 3.0, 4.0, 1.0, 2.0], dtype=cp.float32)
        result = cp.minimum.accumulate(a)
        expected = np.minimum.accumulate(
            np.array([5.0, 3.0, 4.0, 1.0, 2.0], dtype=np.float32)
        )
        assert_allclose(result.get(), expected)

    def test_accumulate_increasing(self):
        a = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32)
        result = cp.minimum.accumulate(a)
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        assert_allclose(result.get(), expected)
