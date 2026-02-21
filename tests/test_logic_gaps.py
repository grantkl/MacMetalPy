"""Tests for logic extras -- iscomplexobj, isrealobj, isfortran.

These are the missing logic functions being added to logic_ops.py.
Ref: numpy.iscomplexobj, numpy.isrealobj, numpy.isfortran
"""

import numpy as np
import pytest

import macmetalpy as cp
from macmetalpy.logic_ops import iscomplexobj, isrealobj, isfortran
from conftest import FLOAT_DTYPES, INT_DTYPES


# ====================================================================
# iscomplexobj
# ====================================================================

class TestIscomplexobj:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_real_array_is_not_complex(self, dtype):
        a = cp.array([1.0, 2.0, 3.0], dtype=dtype)
        assert iscomplexobj(a) is False

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_int_array_is_not_complex(self, dtype):
        a = cp.array([1, 2, 3], dtype=dtype)
        assert iscomplexobj(a) is False

    def test_complex_array_is_complex(self):
        a = cp.array([1 + 2j, 3 + 0j], dtype=np.complex64)
        assert iscomplexobj(a) is True

    def test_bool_array_is_not_complex(self):
        a = cp.array([True, False, True])
        assert iscomplexobj(a) is False

    def test_python_float(self):
        assert iscomplexobj(1.0) is False

    def test_python_complex(self):
        assert iscomplexobj(1 + 2j) is True

    def test_python_int(self):
        assert iscomplexobj(5) is False

    def test_numpy_complex_array(self):
        a = np.array([1 + 2j], dtype=np.complex64)
        assert iscomplexobj(a) is True

    def test_numpy_real_array(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        assert iscomplexobj(a) is False


# ====================================================================
# isrealobj
# ====================================================================

class TestIsrealobj:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_real_array_is_real(self, dtype):
        a = cp.array([1.0, 2.0, 3.0], dtype=dtype)
        assert isrealobj(a) is True

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_int_array_is_real(self, dtype):
        a = cp.array([1, 2, 3], dtype=dtype)
        assert isrealobj(a) is True

    def test_complex_array_is_not_real(self):
        a = cp.array([1 + 2j, 3 + 0j], dtype=np.complex64)
        assert isrealobj(a) is False

    def test_bool_array_is_real(self):
        a = cp.array([True, False, True])
        assert isrealobj(a) is True

    def test_python_float(self):
        assert isrealobj(1.0) is True

    def test_python_complex(self):
        assert isrealobj(1 + 2j) is False

    def test_python_int(self):
        assert isrealobj(5) is True

    def test_numpy_complex_array(self):
        a = np.array([1 + 2j], dtype=np.complex64)
        assert isrealobj(a) is False

    def test_numpy_real_array(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        assert isrealobj(a) is True


# ====================================================================
# isfortran
# ====================================================================

class TestIsfortran:
    def test_c_contiguous_array(self):
        a = cp.zeros((3, 4))
        assert isfortran(a) is False

    def test_1d_array(self):
        a = cp.array([1.0, 2.0, 3.0])
        assert isfortran(a) is False

    def test_scalar_array(self):
        a = cp.array(1.0)
        assert isfortran(a) is False

    def test_always_false(self):
        # Metal is row-major only, so isfortran always returns False
        a = cp.zeros((5, 5))
        assert isfortran(a) is False

    def test_3d_array(self):
        a = cp.zeros((2, 3, 4))
        assert isfortran(a) is False
