"""Tests for FFT functions (CuPy-compatible API).

Rewrite of existing test_fft.py with full parametrization.
Ref: cupy_tests/fft_tests/
Target: ~106 parametrized cases.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy import fft as cpfft

from conftest import FLOAT_DTYPES, tol_for


# ── tolerance for FFT operations ──────────────────────────────────────────
FFT_TOL = dict(rtol=1e-3, atol=1e-4)


# ======================================================================
# 1-D FFT / IFFT
# ======================================================================
# Ref: cupy_tests/fft_tests/test_fft.py

class TestFft:
    def test_power_of_2(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.fft(cp.array(a))
        expected = np.fft.fft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_prime_length(self):
        np.random.seed(42)
        a = np.random.randn(13).astype(np.float32)
        result = cpfft.fft(cp.array(a))
        expected = np.fft.fft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_odd_length(self):
        np.random.seed(42)
        a = np.random.randn(15).astype(np.float32)
        result = cpfft.fft(cp.array(a))
        expected = np.fft.fft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_with_n(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.fft(cp.array(a), n=32)
        expected = np.fft.fft(a, n=32)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_single_element(self):
        a = np.array([3.0], dtype=np.float32)
        result = cpfft.fft(cp.array(a))
        expected = np.fft.fft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_axis(self):
        np.random.seed(42)
        a = np.random.randn(4, 8).astype(np.float32)
        result = cpfft.fft(cp.array(a), axis=0)
        expected = np.fft.fft(a, axis=0)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


class TestIfft:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.ifft(cp.array(a))
        expected = np.fft.ifft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_roundtrip_power_of_2(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        roundtrip = cpfft.ifft(cpfft.fft(cp.array(a)))
        npt.assert_allclose(roundtrip.get().real, a, **FFT_TOL)

    def test_roundtrip_prime(self):
        np.random.seed(42)
        a = np.random.randn(13).astype(np.float32)
        roundtrip = cpfft.ifft(cpfft.fft(cp.array(a)))
        npt.assert_allclose(roundtrip.get().real, a, **FFT_TOL)

    def test_roundtrip_odd(self):
        np.random.seed(42)
        a = np.random.randn(15).astype(np.float32)
        roundtrip = cpfft.ifft(cpfft.fft(cp.array(a)))
        npt.assert_allclose(roundtrip.get().real, a, **FFT_TOL)


# ======================================================================
# 2-D FFT
# ======================================================================

class TestFft2:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        result = cpfft.fft2(cp.array(a))
        expected = np.fft.fft2(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_with_s(self):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        result = cpfft.fft2(cp.array(a), s=(16, 16))
        expected = np.fft.fft2(a, s=(16, 16))
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_roundtrip(self):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        roundtrip = cpfft.ifft2(cpfft.fft2(cp.array(a)))
        npt.assert_allclose(roundtrip.get().real, a, **FFT_TOL)


class TestIfft2:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        result = cpfft.ifft2(cp.array(a))
        expected = np.fft.ifft2(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


# ======================================================================
# N-D FFT
# ======================================================================

class TestFftn:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        result = cpfft.fftn(cp.array(a))
        expected = np.fft.fftn(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_with_s(self):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        result = cpfft.fftn(cp.array(a), s=(8, 8, 8))
        expected = np.fft.fftn(a, s=(8, 8, 8))
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_roundtrip(self):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        roundtrip = cpfft.ifftn(cpfft.fftn(cp.array(a)))
        npt.assert_allclose(roundtrip.get().real, a, **FFT_TOL)


class TestIfftn:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        result = cpfft.ifftn(cp.array(a))
        expected = np.fft.ifftn(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


# ======================================================================
# Real FFT
# ======================================================================

class TestRfft:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.rfft(cp.array(a))
        expected = np.fft.rfft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_with_n(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.rfft(cp.array(a), n=32)
        expected = np.fft.rfft(a, n=32)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_roundtrip(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        roundtrip = cpfft.irfft(cpfft.rfft(cp.array(a)))
        npt.assert_allclose(roundtrip.get(), a, **FFT_TOL)


class TestIrfft:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        freq = np.fft.rfft(a)
        result = cpfft.irfft(cp.array(freq))
        expected = np.fft.irfft(freq)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


# ======================================================================
# 2-D Real FFT
# ======================================================================

class TestRfft2:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        result = cpfft.rfft2(cp.array(a))
        expected = np.fft.rfft2(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_roundtrip(self):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        roundtrip = cpfft.irfft2(cpfft.rfft2(cp.array(a)))
        npt.assert_allclose(roundtrip.get(), a, **FFT_TOL)


class TestIrfft2:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        freq = np.fft.rfft2(a)
        result = cpfft.irfft2(cp.array(freq))
        expected = np.fft.irfft2(freq)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


# ======================================================================
# N-D Real FFT
# ======================================================================

class TestRfftn:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        result = cpfft.rfftn(cp.array(a))
        expected = np.fft.rfftn(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_roundtrip(self):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        roundtrip = cpfft.irfftn(cpfft.rfftn(cp.array(a)))
        npt.assert_allclose(roundtrip.get(), a, **FFT_TOL)


class TestIrfftn:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        freq = np.fft.rfftn(a)
        result = cpfft.irfftn(cp.array(freq))
        expected = np.fft.irfftn(freq)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


# ======================================================================
# Hermitian FFT
# ======================================================================

class TestHfft:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        freq = np.fft.ihfft(a)
        result = cpfft.hfft(cp.array(freq))
        expected = np.fft.hfft(freq)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_roundtrip(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        roundtrip = cpfft.hfft(cpfft.ihfft(cp.array(a)))
        npt.assert_allclose(roundtrip.get(), a, **FFT_TOL)


class TestIhfft:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.ihfft(cp.array(a))
        expected = np.fft.ihfft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


# ======================================================================
# Frequency helpers
# ======================================================================

class TestFftfreq:
    def test_basic(self):
        result = cpfft.fftfreq(8)
        expected = np.fft.fftfreq(8)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_with_d(self):
        result = cpfft.fftfreq(8, d=0.5)
        expected = np.fft.fftfreq(8, d=0.5)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_values_check(self):
        result = cpfft.fftfreq(4, d=1.0)
        expected = np.array([0.0, 0.25, -0.5, -0.25], dtype=np.float64)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_odd_n(self):
        result = cpfft.fftfreq(5)
        expected = np.fft.fftfreq(5)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


class TestRfftfreq:
    def test_basic(self):
        result = cpfft.rfftfreq(8)
        expected = np.fft.rfftfreq(8)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_with_d(self):
        result = cpfft.rfftfreq(8, d=0.5)
        expected = np.fft.rfftfreq(8, d=0.5)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_values_check(self):
        result = cpfft.rfftfreq(8, d=1.0)
        expected = np.fft.rfftfreq(8, d=1.0)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


# ======================================================================
# Shift helpers
# ======================================================================

class TestFftshift:
    def test_1d(self):
        a = np.fft.fftfreq(8).astype(np.float32)
        result = cpfft.fftshift(cp.array(a))
        expected = np.fft.fftshift(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_2d(self):
        np.random.seed(42)
        a = np.random.randn(4, 4).astype(np.float32)
        result = cpfft.fftshift(cp.array(a))
        expected = np.fft.fftshift(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_axes(self):
        np.random.seed(42)
        a = np.random.randn(4, 4).astype(np.float32)
        result = cpfft.fftshift(cp.array(a), axes=0)
        expected = np.fft.fftshift(a, axes=0)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


class TestIfftshift:
    def test_1d(self):
        a = np.fft.fftfreq(8).astype(np.float32)
        shifted = np.fft.fftshift(a)
        result = cpfft.ifftshift(cp.array(shifted))
        expected = np.fft.ifftshift(shifted)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_2d(self):
        np.random.seed(42)
        a = np.random.randn(4, 4).astype(np.float32)
        shifted = np.fft.fftshift(a)
        result = cpfft.ifftshift(cp.array(shifted))
        expected = np.fft.ifftshift(shifted)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_roundtrip(self):
        np.random.seed(42)
        a = np.random.randn(8).astype(np.float32)
        roundtrip = cpfft.ifftshift(cpfft.fftshift(cp.array(a)))
        npt.assert_allclose(roundtrip.get(), a, **FFT_TOL)

    def test_axes(self):
        np.random.seed(42)
        a = np.random.randn(4, 4).astype(np.float32)
        shifted = np.fft.fftshift(a, axes=1)
        result = cpfft.ifftshift(cp.array(shifted), axes=1)
        expected = np.fft.ifftshift(shifted, axes=1)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)
