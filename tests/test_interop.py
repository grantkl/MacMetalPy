"""Tests for Python interop protocols: len(), float(), int(), repr(), str().

Ref: cupy_tests/core_tests/test_ndarray.py (Python protocol tests)
"""

import numpy as np
import pytest

import macmetalpy as cp
from conftest import ALL_DTYPES, make_arg


# =====================================================================
# 1. len()  (~30 cases)
# =====================================================================


class TestLen:
    """len() on 1-D, 2-D, 0-D error."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_len_1d(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        assert len(ga) == len(np_a)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_len_2d(self, dtype):
        np_a = make_arg((2, 3), dtype)
        ga = cp.array(np_a)
        assert len(ga) == len(np_a)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_len_0d_error(self, dtype):
        ga = cp.array(make_arg((), dtype))
        with pytest.raises(TypeError):
            len(ga)


# =====================================================================
# 2. float()  (~20 cases)
# =====================================================================


class TestFloat:
    """float() size-1 ok + size-N error."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_float_size1_ok(self, dtype):
        np_a = make_arg((1,), dtype)
        ga = cp.array(np_a)
        if np.issubdtype(dtype, np.complexfloating):
            pytest.skip("complex types cannot be converted to float")
        expected = float(np_a.ravel()[0])
        result = float(ga)
        assert result == pytest.approx(expected, rel=1e-2)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_float_sizeN_error(self, dtype):
        ga = cp.array(make_arg((5,), dtype))
        with pytest.raises(TypeError):
            float(ga)


# =====================================================================
# 3. int()  (~20 cases)
# =====================================================================


class TestInt:
    """int() size-1 ok + size-N error."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_int_size1_ok(self, dtype):
        np_a = make_arg((1,), dtype)
        ga = cp.array(np_a)
        if np.issubdtype(dtype, np.complexfloating):
            pytest.skip("complex types cannot be converted to int")
        expected = int(np_a.ravel()[0])
        result = int(ga)
        assert result == expected

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_int_sizeN_error(self, dtype):
        ga = cp.array(make_arg((5,), dtype))
        with pytest.raises(TypeError):
            int(ga)


# =====================================================================
# 4. repr() / str()  (~20 cases)
# =====================================================================


class TestReprStr:
    """repr()/str() includes correct values."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_repr_includes_values(self, dtype):
        np_a = make_arg((3,), dtype)
        ga = cp.array(np_a)
        r = repr(ga)
        assert isinstance(r, str)
        assert len(r) > 0
        # Should mention macmetalpy
        assert "macmetalpy" in r or "ndarray" in r

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_str_matches_values(self, dtype):
        np_a = make_arg((3,), dtype)
        ga = cp.array(np_a)
        s = str(ga)
        assert isinstance(s, str)
        assert len(s) > 0
        # str() should produce the same as str(np_a) since it delegates to .get()
        expected_str = str(np_a)
        assert s == expected_str

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_repr_scalar(self, dtype):
        np_a = make_arg((), dtype)
        ga = cp.array(np_a)
        r = repr(ga)
        assert isinstance(r, str)

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=lambda d: np.dtype(d).name)
    def test_str_scalar(self, dtype):
        np_a = make_arg((), dtype)
        ga = cp.array(np_a)
        s = str(ga)
        assert isinstance(s, str)
