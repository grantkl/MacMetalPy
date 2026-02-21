"""Tests for gap-filling creation functions: geomspace, frombuffer, linspace enhancements."""

import numpy as np
import pytest

import macmetalpy as cp
from conftest import FLOAT_DTYPES, NUMERIC_DTYPES, assert_eq


# ====================================================================
# geomspace
# ====================================================================

class TestGeomspace:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        result = cp.geomspace(1, 1000, num=4, dtype=dtype)
        expected = np.geomspace(1, 1000, num=4).astype(dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_num_50(self, dtype):
        result = cp.geomspace(1, 256, num=50, dtype=dtype)
        expected = np.geomspace(1, 256, num=50).astype(dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_num_1(self, dtype):
        result = cp.geomspace(1, 100, num=1, dtype=dtype)
        expected = np.geomspace(1, 100, num=1).astype(dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fractional_endpoints(self, dtype):
        result = cp.geomspace(0.1, 100, num=10, dtype=dtype)
        expected = np.geomspace(0.1, 100, num=10).astype(dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_same_start_stop(self, dtype):
        result = cp.geomspace(5, 5, num=3, dtype=dtype)
        expected = np.geomspace(5, 5, num=3).astype(dtype)
        assert_eq(result, expected, dtype=dtype)

    def test_default_num(self):
        result = cp.geomspace(1, 100)
        expected = np.geomspace(1, 100).astype(np.float32)
        assert result.shape == (50,)
        assert_eq(result, expected, dtype=np.float32)


# ====================================================================
# frombuffer
# ====================================================================

class TestFrombuffer:
    @pytest.mark.parametrize("dtype", [np.float32, np.int32])
    def test_basic(self, dtype):
        np_data = np.array([1, 2, 3, 4, 5], dtype=dtype)
        buf = np_data.tobytes()
        result = cp.frombuffer(buf, dtype=dtype)
        expected = np.frombuffer(buf, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    def test_with_count(self):
        np_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        buf = np_data.tobytes()
        result = cp.frombuffer(buf, dtype=np.float32, count=3)
        expected = np.frombuffer(buf, dtype=np.float32, count=3)
        assert result.shape == (3,)
        assert_eq(result, expected, dtype=np.float32)

    def test_with_offset(self):
        np_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        buf = np_data.tobytes()
        # Skip the first float32 (4 bytes)
        result = cp.frombuffer(buf, dtype=np.float32, offset=4)
        expected = np.frombuffer(buf, dtype=np.float32, offset=4)
        assert_eq(result, expected, dtype=np.float32)

    def test_with_count_and_offset(self):
        np_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        buf = np_data.tobytes()
        result = cp.frombuffer(buf, dtype=np.float32, count=2, offset=8)
        expected = np.frombuffer(buf, dtype=np.float32, count=2, offset=8)
        assert_eq(result, expected, dtype=np.float32)

    def test_bytearray_input(self):
        np_data = np.array([10, 20, 30], dtype=np.int32)
        buf = bytearray(np_data.tobytes())
        result = cp.frombuffer(buf, dtype=np.int32)
        expected = np.frombuffer(buf, dtype=np.int32)
        assert_eq(result, expected, dtype=np.int32)


# ====================================================================
# linspace enhancements: endpoint and retstep
# ====================================================================

class TestLinspaceEndpoint:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_endpoint_true(self, dtype):
        result = cp.linspace(0, 10, num=5, endpoint=True, dtype=dtype)
        expected = np.linspace(0, 10, num=5, endpoint=True, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_endpoint_false(self, dtype):
        result = cp.linspace(0, 10, num=5, endpoint=False, dtype=dtype)
        expected = np.linspace(0, 10, num=5, endpoint=False, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_endpoint_false_num_1(self, dtype):
        result = cp.linspace(0, 10, num=1, endpoint=False, dtype=dtype)
        expected = np.linspace(0, 10, num=1, endpoint=False, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_endpoint_false_large(self, dtype):
        result = cp.linspace(0, 1, num=100, endpoint=False, dtype=dtype)
        expected = np.linspace(0, 1, num=100, endpoint=False, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)


class TestLinspaceRetstep:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_retstep_true(self, dtype):
        result, step = cp.linspace(0, 10, num=5, retstep=True, dtype=dtype)
        expected, exp_step = np.linspace(0, 10, num=5, retstep=True, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)
        assert np.isclose(step, exp_step, rtol=1e-5)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_retstep_false(self, dtype):
        result = cp.linspace(0, 10, num=5, retstep=False, dtype=dtype)
        assert not isinstance(result, tuple)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_retstep_endpoint_false(self, dtype):
        result, step = cp.linspace(0, 10, num=5, endpoint=False, retstep=True, dtype=dtype)
        expected, exp_step = np.linspace(0, 10, num=5, endpoint=False, retstep=True, dtype=dtype)
        assert_eq(result, expected, dtype=dtype)
        assert np.isclose(step, exp_step, rtol=1e-5)

    def test_retstep_num_0(self):
        result, step = cp.linspace(0, 10, num=0, retstep=True)
        expected, exp_step = np.linspace(0, 10, num=0, retstep=True)
        assert result.shape == expected.shape
        assert np.isnan(step) == np.isnan(exp_step)
