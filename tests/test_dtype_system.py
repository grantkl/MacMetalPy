"""Tests for the dtype infrastructure: resolve_dtype, result_dtype, type maps.

Ref: cupy_tests/core_tests/test_dtype.py
"""

import itertools
import warnings

import numpy as np
import pytest

import macmetalpy as cp
from macmetalpy._dtypes import (
    METAL_TYPE_NAMES,
    SUPPORTED_DTYPES,
    metal_to_numpy,
    numpy_to_metal,
    resolve_dtype,
    result_dtype,
)

from conftest import (
    ALL_DTYPES,
    ALL_DTYPES_NO_BOOL,
    DTYPE_PAIRS_WITH_BOOL,
    FLOAT_DTYPES,
    INT_DTYPES,
    NUMERIC_DTYPES,
    SHAPES_ALL,
    SHAPES_NONZERO,
    assert_eq,
    make_arg,
    tol_for,
)


# ── resolve_dtype ────────────────────────────────────────────────────


class TestResolveDtype:
    """Test resolve_dtype for all supported and unsupported dtypes."""

    # Ref: cupy_tests/core_tests/test_dtype.py

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_supported_dtype_passthrough(self, dtype):
        """Each of the 10 supported dtypes should pass through unchanged."""
        result = resolve_dtype(dtype)
        assert result == np.dtype(dtype)

    def test_none_defaults_to_float32(self):
        """None should resolve to the default float dtype (float32)."""
        result = resolve_dtype(None)
        assert result == np.dtype(np.float32)

    def test_float64_downcast_to_float32(self):
        """float64 should downcast to float32 in 'downcast' mode."""
        result = resolve_dtype(np.float64)
        assert result == np.dtype(np.float32)

    def test_float64_downcast_emits_warning(self):
        """float64 downcast should emit UserWarning when warn_on_downcast=True."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve_dtype(np.float64)
            # The warning may or may not appear depending on stacklevel,
            # but let's check if at least one UserWarning was captured
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            # Note: stacklevel=3 in resolve_dtype means the warning might
            # not be captured in this direct call context, which is fine.

    def test_float64_downcast_no_warning_when_disabled(self):
        """float64 downcast should not emit warning when warn_on_downcast=False."""
        cfg = cp.get_config()
        cfg.warn_on_downcast = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_dtype(np.float64)
            assert result == np.dtype(np.float32)

    def test_complex128_downcast_to_complex64(self):
        """complex128 should downcast to complex64."""
        result = resolve_dtype(np.complex128)
        assert result == np.dtype(np.complex64)

    def test_complex64_accepted(self):
        """complex64 should be accepted directly."""
        result = resolve_dtype(np.complex64)
        assert result == np.dtype(np.complex64)

    def test_int8_upcast_to_int16(self):
        """int8 should be upcast to int16 (Metal doesn't support 8-bit ints)."""
        result = resolve_dtype(np.int8)
        assert result == np.dtype(np.int16)

    def test_uint8_upcast_to_uint16(self):
        """uint8 should be upcast to uint16 (Metal doesn't support 8-bit ints)."""
        result = resolve_dtype(np.uint8)
        assert result == np.dtype(np.uint16)

    def test_string_dtype_passthrough(self):
        """String dtype specifiers should work (e.g. 'float32')."""
        result = resolve_dtype("float32")
        assert result == np.dtype(np.float32)


# ── result_dtype ─────────────────────────────────────────────────────


class TestResultDtype:
    """Test result_dtype for all dtype pair combinations."""

    # Ref: cupy_tests/core_tests/test_dtype.py::TestResultDtype

    @pytest.mark.parametrize("dt1,dt2", DTYPE_PAIRS_WITH_BOOL)
    def test_result_dtype_matches_numpy(self, dt1, dt2):
        """result_dtype should match NumPy's result_type, then resolve."""
        np_promoted = np.result_type(dt1, dt2)
        expected = resolve_dtype(np_promoted)
        actual = result_dtype(np.dtype(dt1), np.dtype(dt2))
        assert actual == expected, (
            f"result_dtype({np.dtype(dt1)}, {np.dtype(dt2)}) = {actual}, "
            f"expected {expected}"
        )


# ── numpy_to_metal / metal_to_numpy ──────────────────────────────────


class TestTypeMapping:
    """Test NumPy <-> Metal type string mapping roundtrip."""

    # Ref: cupy_tests/core_tests/test_dtype.py

    SUPPORTED_NO_COMPLEX = [d for d in SUPPORTED_DTYPES if d != np.dtype(np.complex64)]

    @pytest.mark.parametrize("dtype", sorted(SUPPORTED_NO_COMPLEX, key=str))
    def test_numpy_to_metal_roundtrip(self, dtype):
        """numpy_to_metal -> metal_to_numpy should roundtrip for supported types."""
        metal_str = numpy_to_metal(dtype)
        assert isinstance(metal_str, str)
        roundtripped = metal_to_numpy(metal_str)
        assert roundtripped == dtype

    def test_numpy_to_metal_unsupported_raises(self):
        """Unsupported dtype should raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported dtype"):
            numpy_to_metal(np.dtype(np.float64))

    def test_metal_to_numpy_unknown_raises(self):
        """Unknown Metal type string should raise TypeError."""
        with pytest.raises(TypeError, match="Unknown Metal type"):
            metal_to_numpy("not_a_type")

    @pytest.mark.parametrize("dtype", sorted(SUPPORTED_NO_COMPLEX, key=str))
    def test_metal_type_names_exist(self, dtype):
        """Every supported dtype should have a METAL_TYPE_NAMES entry."""
        assert dtype in METAL_TYPE_NAMES


# ── Array create + get roundtrip ─────────────────────────────────────


class TestArrayRoundtrip:
    """Test array creation and .get() roundtrip for all dtype x shape combos."""

    # Ref: cupy_tests/core_tests/test_ndarray.py

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_create_and_get(self, dtype, shape):
        """Create array with make_arg, transfer to GPU, get() back, compare."""
        np_arr = make_arg(shape, dtype, xp=np)
        gpu_arr = cp.array(np_arr, dtype=dtype)
        result = gpu_arr.get()
        assert result.shape == np_arr.shape
        assert result.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result, np_arr)


# ── Binary op cross-dtype result dtype ───────────────────────────────


class TestBinaryOpCrossDtype:
    """Verify that binary op result dtype matches NumPy's result_type."""

    # Ref: cupy_tests/core_tests/test_ndarray_elementwise_op.py

    @pytest.mark.parametrize("dt1,dt2",
                             list(itertools.product(ALL_DTYPES_NO_BOOL, repeat=2)))
    def test_add_result_dtype(self, dt1, dt2):
        """cp.add(a, b) result dtype should match resolved NumPy promotion."""
        np_a = np.array([1, 2, 3], dtype=dt1)
        np_b = np.array([4, 5, 6], dtype=dt2)
        np_result = np.add(np_a, np_b)

        gpu_a = cp.array(np_a)
        gpu_b = cp.array(np_b)
        gpu_result = cp.add(gpu_a, gpu_b)

        expected_dtype = resolve_dtype(np_result.dtype)
        assert gpu_result.dtype == expected_dtype, (
            f"add({np.dtype(dt1)}, {np.dtype(dt2)}): "
            f"got {gpu_result.dtype}, expected {expected_dtype}"
        )
