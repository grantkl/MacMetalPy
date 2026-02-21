"""Tests for extra API functions (linspace, eye, dot, where, clip, concat, stack)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp
from macmetalpy.creation import linspace, eye
from macmetalpy.math_ops import dot, where, clip, concatenate, stack, vstack, hstack


class TestLinspace:
    def test_basic(self):
        result = linspace(0, 1, 5)
        expected = np.linspace(0, 1, 5).astype(np.float32)
        assert_allclose(result.get(), expected)

    def test_dtype(self):
        result = linspace(0, 10, 5, dtype=cp.float32)
        expected = np.linspace(0, 10, 5, dtype=np.float32)
        assert_allclose(result.get(), expected)

    def test_single_element(self):
        result = linspace(5, 5, 1)
        expected = np.linspace(5, 5, 1).astype(np.float32)
        assert_allclose(result.get(), expected)

    def test_many_elements(self):
        result = linspace(0, 100, 1000)
        expected = np.linspace(0, 100, 1000).astype(np.float32)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestEye:
    def test_basic(self):
        result = eye(3)
        expected = np.eye(3, dtype=np.float32)
        assert_allclose(result.get(), expected)

    def test_non_square(self):
        result = eye(3, 4)
        expected = np.eye(3, 4, dtype=np.float32)
        assert_allclose(result.get(), expected)

    def test_dtype(self):
        result = eye(3, dtype=cp.int32)
        expected = np.eye(3, dtype=np.int32)
        assert_array_equal(result.get(), expected)


class TestDot:
    def test_dot_2d(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        b = cp.array([[5.0, 6.0], [7.0, 8.0]], dtype=cp.float32)
        result = dot(a, b)
        expected = np.dot(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        )
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_dot_1d(self):
        """1-D dot product (inner product)."""
        a = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        b = cp.array([4.0, 5.0, 6.0], dtype=cp.float32)
        result = dot(a, b)
        expected = np.dot(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        )
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestWhere:
    def test_basic(self):
        cond = cp.array([True, False, True, False], dtype=cp.bool_)
        x = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        y = cp.array([10.0, 20.0, 30.0, 40.0], dtype=cp.float32)
        result = where(cond, x, y)
        expected = np.where(
            np.array([True, False, True, False]),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        )
        assert_allclose(result.get(), expected)

    def test_scalar_values(self):
        cond = cp.array([True, False, True], dtype=cp.bool_)
        result = where(cond, 1.0, 0.0)
        expected = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        assert_allclose(result.get(), expected)


class TestClip:
    def test_basic(self):
        a = cp.array([-1.0, 0.5, 1.5, 3.0], dtype=cp.float32)
        result = clip(a, 0.0, 1.0)
        expected = np.clip(np.array([-1.0, 0.5, 1.5, 3.0], dtype=np.float32), 0.0, 1.0)
        assert_allclose(result.get(), expected)

    def test_no_clip_needed(self):
        a = cp.array([0.2, 0.5, 0.8], dtype=cp.float32)
        result = clip(a, 0.0, 1.0)
        expected = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        assert_allclose(result.get(), expected)


class TestConcatenate:
    def test_basic(self):
        a = cp.array([1.0, 2.0], dtype=cp.float32)
        b = cp.array([3.0, 4.0], dtype=cp.float32)
        result = concatenate([a, b])
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert_allclose(result.get(), expected)

    def test_three_arrays(self):
        a = cp.array([1.0], dtype=cp.float32)
        b = cp.array([2.0, 3.0], dtype=cp.float32)
        c = cp.array([4.0], dtype=cp.float32)
        result = concatenate([a, b, c])
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert_allclose(result.get(), expected)

    def test_2d_axis0(self):
        a = cp.array([[1.0, 2.0]], dtype=cp.float32)
        b = cp.array([[3.0, 4.0]], dtype=cp.float32)
        result = concatenate([a, b], axis=0)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assert_allclose(result.get(), expected)


class TestStack:
    def test_basic(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        b = cp.array([4.0, 5.0, 6.0], dtype=cp.float32)
        result = stack([a, b])
        expected = np.stack([
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ])
        assert_allclose(result.get(), expected)

    def test_axis1(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        b = cp.array([4.0, 5.0, 6.0], dtype=cp.float32)
        result = stack([a, b], axis=1)
        expected = np.stack([
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ], axis=1)
        assert_allclose(result.get(), expected)


class TestVstackHstack:
    def test_vstack(self):
        a = cp.array([1.0, 2.0], dtype=cp.float32)
        b = cp.array([3.0, 4.0], dtype=cp.float32)
        result = vstack([a, b])
        expected = np.vstack([
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ])
        assert_allclose(result.get(), expected)

    def test_hstack(self):
        a = cp.array([1.0, 2.0], dtype=cp.float32)
        b = cp.array([3.0, 4.0], dtype=cp.float32)
        result = hstack([a, b])
        expected = np.hstack([
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ])
        assert_allclose(result.get(), expected)
