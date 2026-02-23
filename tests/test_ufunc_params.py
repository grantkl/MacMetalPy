"""Tests that ufunc-like functions accept standard ufunc kwargs without raising."""
import numpy as np
import macmetalpy as mp
import pytest

UFUNC_KWARGS = dict(out=None, where=True, casting='same_kind', subok=True, dtype=None, order='K', signature=None)

# List ALL ufunc-like functions to test
UNARY_UFUNCS = [
    'abs', 'absolute', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh',
    'bitwise_not', 'cbrt', 'ceil', 'conj', 'conjugate', 'cos', 'cosh',
    'deg2rad', 'degrees', 'exp', 'exp2', 'expm1', 'fabs', 'floor',
    'invert', 'isfinite', 'isinf', 'isnan', 'log', 'log10', 'log1p', 'log2',
    'logical_not', 'negative', 'positive', 'rad2deg', 'radians', 'reciprocal',
    'rint', 'sign', 'signbit', 'sin', 'sinh', 'sqrt', 'square', 'tan', 'tanh',
    'trunc',
]

BINARY_UFUNCS = [
    'add', 'arctan2', 'bitwise_and', 'bitwise_left_shift', 'bitwise_or',
    'bitwise_right_shift', 'bitwise_xor', 'copysign', 'divide',
    'equal', 'float_power', 'floor_divide', 'fmax', 'fmin', 'fmod',
    'greater', 'greater_equal', 'heaviside', 'hypot', 'lcm', 'gcd',
    'ldexp', 'left_shift', 'less', 'less_equal', 'logaddexp', 'logaddexp2',
    'logical_and', 'logical_or', 'logical_xor', 'maximum', 'minimum',
    'mod', 'multiply', 'nextafter', 'not_equal', 'power', 'remainder',
    'right_shift', 'subtract', 'true_divide',
]

@pytest.mark.parametrize("name", UNARY_UFUNCS)
def test_unary_ufunc_accepts_kwargs(name):
    fn = getattr(mp, name, None)
    if fn is None:
        pytest.skip(f"mp.{name} not found")
    a = mp.array([1.0, 2.0, 3.0])
    # Should not raise when passing ufunc kwargs
    try:
        fn(a, **UFUNC_KWARGS)
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            pytest.fail(f"mp.{name} does not accept ufunc kwargs: {e}")
        # Other TypeErrors (e.g., domain issues) are fine

@pytest.mark.parametrize("name", BINARY_UFUNCS)
def test_binary_ufunc_accepts_kwargs(name):
    fn = getattr(mp, name, None)
    if fn is None:
        pytest.skip(f"mp.{name} not found")
    a = mp.array([1.0, 2.0, 3.0])
    b = mp.array([4.0, 5.0, 6.0])
    try:
        fn(a, b, **UFUNC_KWARGS)
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            pytest.fail(f"mp.{name} does not accept ufunc kwargs: {e}")
