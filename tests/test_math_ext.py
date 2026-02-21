"""Tests for extended math functions (sign, floor, ceil, trig, log2, log10, etc.)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp


@pytest.fixture
def positive_floats():
    np_arr = np.array([0.1, 0.5, 1.0, 2.0, 3.5], dtype=np.float32)
    return cp.array(np_arr), np_arr


@pytest.fixture
def mixed_floats():
    np_arr = np.array([-3.7, -1.2, 0.0, 1.2, 3.7], dtype=np.float32)
    return cp.array(np_arr), np_arr


@pytest.fixture
def trig_inputs():
    """Values in [-1, 1] for arcsin/arccos, general for sin/cos/tan."""
    np_arr = np.array([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=np.float32)
    return cp.array(np_arr), np_arr


@pytest.fixture
def general_floats():
    np_arr = np.array([-2.0, -1.0, 0.5, 1.0, 2.0], dtype=np.float32)
    return cp.array(np_arr), np_arr


class TestSign:
    def test_sign_positive(self):
        np_arr = np.array([1.0, 2.5, 100.0], dtype=np.float32)
        result = cp.sign(cp.array(np_arr))
        expected = np.sign(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_sign_negative(self):
        np_arr = np.array([-1.0, -2.5, -100.0], dtype=np.float32)
        result = cp.sign(cp.array(np_arr))
        expected = np.sign(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_sign_zero(self):
        np_arr = np.array([0.0, 0.0], dtype=np.float32)
        result = cp.sign(cp.array(np_arr))
        expected = np.sign(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_sign_mixed(self, mixed_floats):
        gpu, np_arr = mixed_floats
        result = cp.sign(gpu)
        expected = np.sign(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestFloor:
    def test_floor_positive(self, positive_floats):
        gpu, np_arr = positive_floats
        result = cp.floor(gpu)
        expected = np.floor(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_floor_negative(self):
        np_arr = np.array([-1.1, -2.9, -0.5], dtype=np.float32)
        result = cp.floor(cp.array(np_arr))
        expected = np.floor(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_floor_mixed(self, mixed_floats):
        gpu, np_arr = mixed_floats
        result = cp.floor(gpu)
        expected = np.floor(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestCeil:
    def test_ceil_positive(self, positive_floats):
        gpu, np_arr = positive_floats
        result = cp.ceil(gpu)
        expected = np.ceil(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_ceil_negative(self):
        np_arr = np.array([-1.1, -2.9, -0.5], dtype=np.float32)
        result = cp.ceil(cp.array(np_arr))
        expected = np.ceil(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_ceil_mixed(self, mixed_floats):
        gpu, np_arr = mixed_floats
        result = cp.ceil(gpu)
        expected = np.ceil(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestSin:
    def test_sin_basic(self, general_floats):
        gpu, np_arr = general_floats
        result = cp.sin(gpu)
        expected = np.sin(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_sin_zero(self):
        np_arr = np.array([0.0], dtype=np.float32)
        result = cp.sin(cp.array(np_arr))
        assert_allclose(result.get(), np.sin(np_arr), rtol=1e-5)


class TestCos:
    def test_cos_basic(self, general_floats):
        gpu, np_arr = general_floats
        result = cp.cos(gpu)
        expected = np.cos(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_cos_zero(self):
        np_arr = np.array([0.0], dtype=np.float32)
        result = cp.cos(cp.array(np_arr))
        assert_allclose(result.get(), np.cos(np_arr), rtol=1e-5)


class TestTan:
    def test_tan_basic(self):
        np_arr = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        result = cp.tan(cp.array(np_arr))
        expected = np.tan(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestArcsin:
    def test_arcsin_basic(self, trig_inputs):
        gpu, np_arr = trig_inputs
        result = cp.arcsin(gpu)
        expected = np.arcsin(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_arcsin_zero(self):
        np_arr = np.array([0.0], dtype=np.float32)
        result = cp.arcsin(cp.array(np_arr))
        assert_allclose(result.get(), np.arcsin(np_arr), rtol=1e-5)


class TestArccos:
    def test_arccos_basic(self, trig_inputs):
        gpu, np_arr = trig_inputs
        result = cp.arccos(gpu)
        expected = np.arccos(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_arccos_one(self):
        np_arr = np.array([1.0], dtype=np.float32)
        result = cp.arccos(cp.array(np_arr))
        assert_allclose(result.get(), np.arccos(np_arr), rtol=1e-5)


class TestArctan:
    def test_arctan_basic(self, general_floats):
        gpu, np_arr = general_floats
        result = cp.arctan(gpu)
        expected = np.arctan(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_arctan_zero(self):
        np_arr = np.array([0.0], dtype=np.float32)
        result = cp.arctan(cp.array(np_arr))
        assert_allclose(result.get(), np.arctan(np_arr), rtol=1e-5)


class TestSinh:
    def test_sinh_basic(self, general_floats):
        gpu, np_arr = general_floats
        result = cp.sinh(gpu)
        expected = np.sinh(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestCosh:
    def test_cosh_basic(self, general_floats):
        gpu, np_arr = general_floats
        result = cp.cosh(gpu)
        expected = np.cosh(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestTanh:
    def test_tanh_basic(self, general_floats):
        gpu, np_arr = general_floats
        result = cp.tanh(gpu)
        expected = np.tanh(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_tanh_zero(self):
        np_arr = np.array([0.0], dtype=np.float32)
        result = cp.tanh(cp.array(np_arr))
        assert_allclose(result.get(), np.tanh(np_arr), rtol=1e-5)


class TestLog2:
    def test_log2_basic(self, positive_floats):
        gpu, np_arr = positive_floats
        result = cp.log2(gpu)
        expected = np.log2(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_log2_powers_of_two(self):
        np_arr = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        result = cp.log2(cp.array(np_arr))
        expected = np.log2(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestLog10:
    def test_log10_basic(self, positive_floats):
        gpu, np_arr = positive_floats
        result = cp.log10(gpu)
        expected = np.log10(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_log10_powers_of_ten(self):
        np_arr = np.array([1.0, 10.0, 100.0, 1000.0], dtype=np.float32)
        result = cp.log10(cp.array(np_arr))
        expected = np.log10(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestSquare:
    def test_square_positive(self, positive_floats):
        gpu, np_arr = positive_floats
        result = cp.square(gpu)
        expected = np.square(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_square_negative(self):
        np_arr = np.array([-3.0, -2.0, -1.0], dtype=np.float32)
        result = cp.square(cp.array(np_arr))
        expected = np.square(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_square_zero(self):
        np_arr = np.array([0.0], dtype=np.float32)
        result = cp.square(cp.array(np_arr))
        assert_allclose(result.get(), np.square(np_arr), rtol=1e-5)


class TestNegative:
    def test_negative_basic(self, mixed_floats):
        gpu, np_arr = mixed_floats
        result = cp.negative(gpu)
        expected = np.negative(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_negative_zero(self):
        np_arr = np.array([0.0], dtype=np.float32)
        result = cp.negative(cp.array(np_arr))
        expected = np.negative(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestAround:
    def test_around_default(self):
        np_arr = np.array([1.4, 1.5, 2.5, 3.6], dtype=np.float32)
        result = cp.around(cp.array(np_arr))
        expected = np.around(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_around_decimals(self):
        np_arr = np.array([1.234, 2.567, 3.891], dtype=np.float32)
        result = cp.around(cp.array(np_arr), decimals=2)
        expected = np.around(np_arr, decimals=2)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_around_negative_decimals(self):
        np_arr = np.array([123.0, 456.0, 789.0], dtype=np.float32)
        result = cp.around(cp.array(np_arr), decimals=-1)
        expected = np.around(np_arr, decimals=-1)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestRound:
    def test_round_alias(self):
        np_arr = np.array([1.4, 1.5, 2.5, 3.6], dtype=np.float32)
        gpu = cp.array(np_arr)
        result_around = cp.around(gpu)
        result_round = cp.round_(gpu)
        assert_allclose(result_round.get(), result_around.get(), rtol=1e-5)


class TestMod:
    def test_mod_basic(self):
        a_np = np.array([7.0, 8.0, 9.0], dtype=np.float32)
        b_np = np.array([3.0, 3.0, 4.0], dtype=np.float32)
        result = cp.mod(cp.array(a_np), cp.array(b_np))
        expected = np.mod(a_np, b_np)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_mod_negative(self):
        a_np = np.array([-7.0, 8.0, -9.0], dtype=np.float32)
        b_np = np.array([3.0, -3.0, -4.0], dtype=np.float32)
        result = cp.mod(cp.array(a_np), cp.array(b_np))
        expected = np.mod(a_np, b_np)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestRemainder:
    def test_remainder_alias(self):
        a_np = np.array([7.0, 8.0, 9.0], dtype=np.float32)
        b_np = np.array([3.0, 3.0, 4.0], dtype=np.float32)
        result_mod = cp.mod(cp.array(a_np), cp.array(b_np))
        result_rem = cp.remainder(cp.array(a_np), cp.array(b_np))
        assert_allclose(result_rem.get(), result_mod.get(), rtol=1e-5)
