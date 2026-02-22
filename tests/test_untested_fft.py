"""Tests for untested FFT APIs: hfft, ihfft, rfftfreq, ifftshift.

Verifies macmetalpy results match numpy.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy import fft as cpfft


FFT_TOL = dict(rtol=1e-3, atol=1e-4)


# ======================================================================
# hfft
# ======================================================================

class TestHfft:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.hfft(cp.array(a))
        expected = np.fft.hfft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_with_n(self):
        np.random.seed(42)
        a = np.random.randn(8).astype(np.float32)
        result = cpfft.hfft(cp.array(a), n=16)
        expected = np.fft.hfft(a, n=16)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_output_is_real(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.hfft(cp.array(a))
        assert np.isrealobj(result.get())

    def test_output_length_default(self):
        a = np.random.randn(9).astype(np.float32)
        result = cpfft.hfft(cp.array(a))
        expected = np.fft.hfft(a)
        assert result.shape == expected.shape

    def test_complex_input(self):
        np.random.seed(42)
        a = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
        result = cpfft.hfft(cp.array(a))
        expected = np.fft.hfft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_2d_axis0(self):
        np.random.seed(42)
        a = np.random.randn(4, 8).astype(np.float32)
        result = cpfft.hfft(cp.array(a), axis=0)
        expected = np.fft.hfft(a, axis=0)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_2d_axis1(self):
        np.random.seed(42)
        a = np.random.randn(4, 8).astype(np.float32)
        result = cpfft.hfft(cp.array(a), axis=1)
        expected = np.fft.hfft(a, axis=1)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


# ======================================================================
# ihfft
# ======================================================================

class TestIhfft:
    def test_basic(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.ihfft(cp.array(a))
        expected = np.fft.ihfft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_with_n(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.ihfft(cp.array(a), n=32)
        expected = np.fft.ihfft(a, n=32)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_output_length(self):
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.ihfft(cp.array(a))
        expected = np.fft.ihfft(a)
        assert result.shape == expected.shape

    def test_roundtrip_with_hfft(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        # hfft(ihfft(a)) should be close to a (up to length changes)
        intermediate = cpfft.ihfft(cp.array(a))
        result = cpfft.hfft(intermediate, n=16)
        npt.assert_allclose(result.get(), a, rtol=1e-2, atol=1e-2)

    def test_2d_axis0(self):
        np.random.seed(42)
        a = np.random.randn(4, 8).astype(np.float32)
        result = cpfft.ihfft(cp.array(a), axis=0)
        expected = np.fft.ihfft(a, axis=0)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_even_length(self):
        np.random.seed(42)
        a = np.random.randn(8).astype(np.float32)
        result = cpfft.ihfft(cp.array(a))
        expected = np.fft.ihfft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_odd_length(self):
        np.random.seed(42)
        a = np.random.randn(7).astype(np.float32)
        result = cpfft.ihfft(cp.array(a))
        expected = np.fft.ihfft(a)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)


# ======================================================================
# rfftfreq
# ======================================================================

class TestRfftfreq:
    def test_basic(self):
        result = cpfft.rfftfreq(8)
        expected = np.fft.rfftfreq(8)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_with_d(self):
        result = cpfft.rfftfreq(8, d=0.5)
        expected = np.fft.rfftfreq(8, d=0.5)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_odd_n(self):
        result = cpfft.rfftfreq(7)
        expected = np.fft.rfftfreq(7)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_output_length_even(self):
        n = 16
        result = cpfft.rfftfreq(n)
        assert result.shape == (n // 2 + 1,)

    def test_output_length_odd(self):
        n = 15
        result = cpfft.rfftfreq(n)
        assert result.shape == ((n + 1) // 2,)

    def test_large_n(self):
        result = cpfft.rfftfreq(1024)
        expected = np.fft.rfftfreq(1024)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_values_range(self):
        result = cpfft.rfftfreq(8).get()
        assert result.min() >= 0.0
        assert result.max() <= 0.5


# ======================================================================
# ifftshift
# ======================================================================

class TestIfftshift:
    def test_1d(self):
        a = np.array([0, 1, 2, 3, 4, -4, -3, -2, -1], dtype=np.float32)
        shifted = np.fft.fftshift(a)
        result = cpfft.ifftshift(cp.array(shifted))
        expected = np.fft.ifftshift(shifted)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_roundtrip_with_fftshift(self):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        shifted = cpfft.ifftshift(cp.array(np.fft.fftshift(a)))
        npt.assert_allclose(shifted.get(), a, **FFT_TOL)

    def test_2d(self):
        np.random.seed(42)
        a = np.random.randn(4, 6).astype(np.float32)
        shifted = np.fft.fftshift(a)
        result = cpfft.ifftshift(cp.array(shifted))
        expected = np.fft.ifftshift(shifted)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_2d_axes0(self):
        np.random.seed(42)
        a = np.random.randn(4, 6).astype(np.float32)
        shifted = np.fft.fftshift(a, axes=0)
        result = cpfft.ifftshift(cp.array(shifted), axes=0)
        expected = np.fft.ifftshift(shifted, axes=0)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_even_length(self):
        a = np.arange(8, dtype=np.float32)
        result = cpfft.ifftshift(cp.array(np.fft.fftshift(a)))
        npt.assert_allclose(result.get(), a, **FFT_TOL)

    def test_odd_length(self):
        a = np.arange(7, dtype=np.float32)
        result = cpfft.ifftshift(cp.array(np.fft.fftshift(a)))
        npt.assert_allclose(result.get(), a, **FFT_TOL)

    def test_preserves_shape(self):
        np.random.seed(42)
        a = np.random.randn(3, 5).astype(np.float32)
        result = cpfft.ifftshift(cp.array(a))
        assert result.shape == a.shape
