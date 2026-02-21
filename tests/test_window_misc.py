"""Tests for window functions and extended math functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp
from macmetalpy.window import bartlett, blackman, hamming, hanning, kaiser
from macmetalpy.math_ext import sinc, i0, convolve, interp, fix, unwrap, trapezoid


# ── Window functions ──────────────────────────────────────────────────────

class TestBartlett:
    def test_basic(self):
        result = bartlett(10)
        expected = np.bartlett(10).astype(np.float32)
        assert result.shape == (10,)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_one(self):
        result = bartlett(1)
        expected = np.bartlett(1).astype(np.float32)
        assert_allclose(result.get(), expected)

    def test_large(self):
        result = bartlett(100)
        expected = np.bartlett(100).astype(np.float32)
        assert result.shape == (100,)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestBlackman:
    def test_basic(self):
        result = blackman(10)
        expected = np.blackman(10).astype(np.float32)
        assert result.shape == (10,)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_one(self):
        result = blackman(1)
        expected = np.blackman(1).astype(np.float32)
        assert_allclose(result.get(), expected)


class TestHamming:
    def test_basic(self):
        result = hamming(10)
        expected = np.hamming(10).astype(np.float32)
        assert result.shape == (10,)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_one(self):
        result = hamming(1)
        expected = np.hamming(1).astype(np.float32)
        assert_allclose(result.get(), expected)


class TestHanning:
    def test_basic(self):
        result = hanning(10)
        expected = np.hanning(10).astype(np.float32)
        assert result.shape == (10,)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_one(self):
        result = hanning(1)
        expected = np.hanning(1).astype(np.float32)
        assert_allclose(result.get(), expected)


class TestKaiser:
    def test_basic(self):
        result = kaiser(10, 5.0)
        expected = np.kaiser(10, 5.0).astype(np.float32)
        assert result.shape == (10,)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_different_beta(self):
        result = kaiser(20, 14.0)
        expected = np.kaiser(20, 14.0).astype(np.float32)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_one(self):
        result = kaiser(1, 0.0)
        expected = np.kaiser(1, 0.0).astype(np.float32)
        assert_allclose(result.get(), expected)


# ── Extended math functions ───────────────────────────────────────────────

class TestSinc:
    def test_basic(self):
        vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        x = cp.array(vals)
        expected = np.sinc(vals)
        assert_allclose(sinc(x).get(), expected, atol=1e-6)

    def test_negative(self):
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        x = cp.array(vals)
        expected = np.sinc(vals)
        assert_allclose(sinc(x).get(), expected, atol=1e-6)


class TestI0:
    def test_basic(self):
        x = cp.array([0.0, 1.0, 2.0, 3.0])
        expected = np.i0(np.array([0.0, 1.0, 2.0, 3.0]))
        assert_allclose(i0(x).get(), expected.astype(np.float64), rtol=1e-5)

    def test_single(self):
        x = cp.array([0.0])
        result = i0(x)
        assert_allclose(result.get(), np.array([1.0]), rtol=1e-5)


class TestConvolve:
    def test_full(self):
        a = cp.array([1.0, 2.0, 3.0])
        v = cp.array([0.0, 1.0, 0.5])
        expected = np.convolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5])
        assert_allclose(convolve(a, v).get(), expected.astype(np.float32), rtol=1e-5)

    def test_same(self):
        a = cp.array([1.0, 2.0, 3.0, 4.0])
        v = cp.array([1.0, 1.0])
        expected = np.convolve([1.0, 2.0, 3.0, 4.0], [1.0, 1.0], mode='same')
        assert_allclose(convolve(a, v, mode='same').get(), expected.astype(np.float32), rtol=1e-5)

    def test_valid(self):
        a = cp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = cp.array([1.0, 2.0])
        expected = np.convolve([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0], mode='valid')
        assert_allclose(convolve(a, v, mode='valid').get(), expected.astype(np.float32), rtol=1e-5)


class TestInterp:
    def test_basic(self):
        xp = cp.array([1.0, 2.0, 3.0])
        fp = cp.array([3.0, 2.0, 0.0])
        x = cp.array([0.0, 1.0, 1.5, 2.72, 3.14])
        expected = np.interp([0.0, 1.0, 1.5, 2.72, 3.14], [1.0, 2.0, 3.0], [3.0, 2.0, 0.0])
        assert_allclose(interp(x, xp, fp).get(), expected.astype(np.float64), rtol=1e-5)

    def test_left_right(self):
        xp = cp.array([1.0, 2.0, 3.0])
        fp = cp.array([3.0, 2.0, 0.0])
        x = cp.array([0.0, 4.0])
        expected = np.interp([0.0, 4.0], [1.0, 2.0, 3.0], [3.0, 2.0, 0.0], left=-1.0, right=-1.0)
        assert_allclose(interp(x, xp, fp, left=-1.0, right=-1.0).get(), expected.astype(np.float64), rtol=1e-5)


class TestFix:
    def test_basic(self):
        x = cp.array([2.1, 2.9, -2.1, -2.9])
        expected = np.fix(np.array([2.1, 2.9, -2.1, -2.9]))
        assert_allclose(fix(x).get(), expected.astype(np.float32), rtol=1e-5)

    def test_zero(self):
        x = cp.array([0.0])
        assert_allclose(fix(x).get(), np.array([0.0], dtype=np.float32))


class TestUnwrap:
    def test_basic(self):
        phase = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        x = cp.array(phase)
        expected = np.unwrap(phase)
        assert_allclose(unwrap(x).get(), expected.astype(np.float32), rtol=1e-5)

    def test_large_jump(self):
        phase = np.array([0.0, np.pi, 2 * np.pi, 3 * np.pi], dtype=np.float32)
        x = cp.array(phase)
        expected = np.unwrap(phase)
        assert_allclose(unwrap(x).get(), expected.astype(np.float32), rtol=1e-4)


class TestTrapezoid:
    def test_basic(self):
        y = cp.array([1.0, 2.0, 3.0, 4.0])
        _trap = getattr(np, 'trapezoid', np.trapz)
        expected = _trap(np.array([1.0, 2.0, 3.0, 4.0]))
        result = trapezoid(y)
        assert_allclose(result.get(), np.array(expected), rtol=1e-5)

    def test_with_x(self):
        y = cp.array([1.0, 2.0, 3.0])
        x = cp.array([0.0, 1.0, 3.0])
        _trap = getattr(np, 'trapezoid', np.trapz)
        expected = _trap(np.array([1.0, 2.0, 3.0]), x=np.array([0.0, 1.0, 3.0]))
        result = trapezoid(y, x=x)
        assert_allclose(result.get(), np.array(expected), rtol=1e-5)

    def test_with_dx(self):
        y = cp.array([1.0, 2.0, 3.0])
        _trap = getattr(np, 'trapezoid', np.trapz)
        expected = _trap(np.array([1.0, 2.0, 3.0]), dx=0.5)
        result = trapezoid(y, dx=0.5)
        assert_allclose(result.get(), np.array(expected), rtol=1e-5)
