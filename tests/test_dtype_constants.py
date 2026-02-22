"""Tests for dtype aliases, constants, and utility re-exports."""

import math

import numpy as np
import pytest

import macmetalpy as cp


# ---------------------------------------------------------------------------
# 1. Dtype alias tests (parametrized)
# ---------------------------------------------------------------------------

DTYPE_ALIASES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float64",
    "bool_",
    "byte",
    "ubyte",
    "short",
    "ushort",
    "intc",
    "uintc",
    "uint",
    "longlong",
    "ulonglong",
    "single",
    "longdouble",
    "complex128",
    "csingle",
    "cdouble",
    "clongdouble",
]

# Aliases removed in NumPy 2, mapped to their underlying type in macmetalpy.
_COMPAT_ALIASES = {
    "longfloat": np.longdouble,
    "clongfloat": np.clongdouble,
    "singlecomplex": np.complex64,
}


@pytest.mark.parametrize("name", DTYPE_ALIASES)
def test_dtype_alias_identity(name):
    """Each dtype alias in macmetalpy should be the exact numpy type."""
    assert getattr(cp, name) is getattr(np, name)


@pytest.mark.parametrize("name,expected", list(_COMPAT_ALIASES.items()))
def test_compat_dtype_alias(name, expected):
    """Compat aliases removed in NumPy 2 should map to the correct type."""
    assert getattr(cp, name) is expected


# ---------------------------------------------------------------------------
# 2. Constant tests
# ---------------------------------------------------------------------------


def test_inf():
    assert cp.inf == float("inf")


def test_pi():
    assert cp.pi == math.pi


def test_newaxis():
    assert cp.newaxis is None


def test_little_endian():
    assert cp.little_endian == np.little_endian


# ---------------------------------------------------------------------------
# 3. Utility re-export tests
# ---------------------------------------------------------------------------


def test_scalar_type():
    assert cp.ScalarType is np.ScalarType


def test_typecodes():
    assert cp.typecodes is np.typecodes


def test_flatiter_is_type():
    assert isinstance(cp.flatiter, type)


# ---------------------------------------------------------------------------
# 4. Alias identity tests
# ---------------------------------------------------------------------------


def test_bitwise_left_shift_alias():
    assert cp.bitwise_left_shift is cp.left_shift


def test_bitwise_right_shift_alias():
    assert cp.bitwise_right_shift is cp.right_shift


def test_random_ranf_callable():
    assert callable(cp.random.ranf)


def test_random_sample_callable():
    assert callable(cp.random.sample)
