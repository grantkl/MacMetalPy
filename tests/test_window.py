"""Tests for window functions.

Extends test_window_misc.py with full parametrization.
Ref: NumPy window function tests
Target: ~26 parametrized cases.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp


# ── shared M values ───────────────────────────────────────────────────────

M_VALUES = [0, 1, 5, 64]


# ======================================================================
# bartlett
# ======================================================================

class TestBartlett:
    @pytest.mark.parametrize("M", M_VALUES)
    def test_values(self, M):
        result = cp.bartlett(M)
        expected = np.bartlett(M).astype(np.float32)
        assert result.shape == (M,)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_symmetry(self):
        result = cp.bartlett(64).get()
        npt.assert_allclose(result, result[::-1], rtol=1e-5)


# ======================================================================
# blackman
# ======================================================================

class TestBlackman:
    @pytest.mark.parametrize("M", M_VALUES)
    def test_values(self, M):
        result = cp.blackman(M)
        expected = np.blackman(M).astype(np.float32)
        assert result.shape == (M,)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_symmetry(self):
        result = cp.blackman(64).get()
        npt.assert_allclose(result, result[::-1], rtol=1e-5)


# ======================================================================
# hamming
# ======================================================================

class TestHamming:
    @pytest.mark.parametrize("M", M_VALUES)
    def test_values(self, M):
        result = cp.hamming(M)
        expected = np.hamming(M).astype(np.float32)
        assert result.shape == (M,)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_symmetry(self):
        result = cp.hamming(64).get()
        npt.assert_allclose(result, result[::-1], rtol=1e-5)


# ======================================================================
# hanning
# ======================================================================

class TestHanning:
    @pytest.mark.parametrize("M", M_VALUES)
    def test_values(self, M):
        result = cp.hanning(M)
        expected = np.hanning(M).astype(np.float32)
        assert result.shape == (M,)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_symmetry(self):
        result = cp.hanning(64).get()
        npt.assert_allclose(result, result[::-1], rtol=1e-5)


# ======================================================================
# kaiser
# ======================================================================

class TestKaiser:
    @pytest.mark.parametrize("M", M_VALUES)
    def test_default_beta(self, M):
        result = cp.kaiser(M, 5.0)
        expected = np.kaiser(M, 5.0).astype(np.float32)
        assert result.shape == (M,)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    @pytest.mark.parametrize("beta", [0.0, 5.0, 14.0])
    def test_beta_variants(self, beta):
        M = 64
        result = cp.kaiser(M, beta)
        expected = np.kaiser(M, beta).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_symmetry(self):
        result = cp.kaiser(64, 5.0).get()
        npt.assert_allclose(result, result[::-1], rtol=1e-5)
