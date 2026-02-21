"""Tests for complex number operations: angle, real, imag, conj, conjugate, real_if_close."""

import numpy as np
import pytest

import macmetalpy as cp
from conftest import FLOAT_DTYPES, assert_eq


# ====================================================================
# angle
# ====================================================================

class TestAngle:
    def test_basic(self):
        z = np.array([1+1j, 1+0j, 0+1j, -1+0j], dtype=np.complex64)
        result = cp.angle(cp.array(z))
        expected = np.angle(z)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)

    def test_deg(self):
        z = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=np.complex64)
        result = cp.angle(cp.array(z), deg=True)
        expected = np.angle(z, deg=True)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)

    def test_real_input(self):
        x = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        result = cp.angle(cp.array(x))
        expected = np.angle(x)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)

    def test_single_value(self):
        z = np.array(1+1j, dtype=np.complex64)
        result = cp.angle(cp.array(z))
        expected = np.angle(z)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)


# ====================================================================
# real
# ====================================================================

class TestReal:
    def test_complex_input(self):
        z = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
        result = cp.real(cp.array(z))
        expected = np.real(z)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)

    def test_real_input(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.real(cp.array(x))
        expected = np.real(x)
        assert_eq(result, expected, dtype=np.float32)

    def test_2d_complex(self):
        z = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
        result = cp.real(cp.array(z))
        expected = np.real(z)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)

    def test_scalar_complex(self):
        z = np.array(3+4j, dtype=np.complex64)
        result = cp.real(cp.array(z))
        expected = np.real(z)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)


# ====================================================================
# imag
# ====================================================================

class TestImag:
    def test_complex_input(self):
        z = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
        result = cp.imag(cp.array(z))
        expected = np.imag(z)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)

    def test_real_input(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.imag(cp.array(x))
        expected = np.imag(x)
        assert_eq(result, expected, dtype=np.float32)

    def test_2d_complex(self):
        z = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
        result = cp.imag(cp.array(z))
        expected = np.imag(z)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)


# ====================================================================
# conj / conjugate
# ====================================================================

class TestConj:
    def test_complex_input(self):
        z = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
        result = cp.conj(cp.array(z))
        expected = np.conj(z)
        assert_eq(result, expected, dtype=np.complex64)

    def test_real_input(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.conj(cp.array(x))
        expected = np.conj(x)
        assert_eq(result, expected, dtype=np.float32)

    def test_2d(self):
        z = np.array([[1+2j, 3-4j], [-5+6j, 7+8j]], dtype=np.complex64)
        result = cp.conj(cp.array(z))
        expected = np.conj(z)
        assert_eq(result, expected, dtype=np.complex64)

    def test_conjugate_alias(self):
        z = np.array([1+2j, 3+4j], dtype=np.complex64)
        result = cp.conjugate(cp.array(z))
        expected = np.conjugate(z)
        assert_eq(result, expected, dtype=np.complex64)


# ====================================================================
# real_if_close
# ====================================================================

class TestRealIfClose:
    def test_close_to_real(self):
        z = np.array([1+1e-15j, 2+0j, 3-1e-15j], dtype=np.complex64)
        result = cp.real_if_close(cp.array(z))
        expected = np.real_if_close(z)
        # Should return real values since imag parts are tiny
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)

    def test_not_close_to_real(self):
        z = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
        result = cp.real_if_close(cp.array(z))
        expected = np.real_if_close(z)
        # Should remain complex
        assert_eq(result, expected, dtype=np.complex64)

    def test_custom_tol(self):
        z = np.array([1+0.01j, 2+0.01j], dtype=np.complex64)
        # With large enough tol, should return real
        result = cp.real_if_close(cp.array(z), tol=1e10)
        expected = np.real_if_close(z, tol=1e10)
        assert_eq(result, expected.astype(np.float32), dtype=np.float32)

    def test_real_input(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.real_if_close(cp.array(x))
        expected = np.real_if_close(x)
        assert_eq(result, expected, dtype=np.float32)


# ====================================================================
# round alias in math_ops
# ====================================================================

class TestRoundAlias:
    def test_round_exists(self):
        assert hasattr(cp, 'round')

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_round_matches_around(self, dtype):
        a_np = np.array([1.567, 2.345, 3.789], dtype=dtype)
        result = cp.round(cp.array(a_np), decimals=2)
        expected = np.around(a_np, decimals=2)
        assert_eq(result, expected, dtype=dtype)
