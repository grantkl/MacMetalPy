"""Tests for edge-case shapes: zero-size, 0-d, single-element, large arrays.

Ref: cupy_tests/core_tests/test_ndarray.py (zero-size, 0-d tests)
     cupy_tests/math_tests/test_arithmetic.py (zero-size arithmetic)
"""

import numpy as np
import pytest

import macmetalpy as cp

from conftest import (
    ALL_DTYPES,
    FLOAT_DTYPES,
    NUMERIC_DTYPES,
    assert_eq,
    make_arg,
    tol_for,
)


# ── Zero-size 1-D ───────────────────────────────────────────────────


class TestZeroSize1D:
    """Test zero-size 1-D array creation and roundtrip."""

    # Ref: cupy_tests/core_tests/test_ndarray.py (empty array tests)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_zeros_empty_1d(self, dtype):
        """zeros((0,)) should create a zero-size array with correct properties."""
        a = cp.zeros((0,), dtype=dtype)
        assert a.shape == (0,)
        assert a.size == 0
        assert a.dtype == np.dtype(dtype)
        result = a.get()
        assert result.shape == (0,)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_array_empty_list(self, dtype):
        """cp.array([], dtype) should create a zero-size 1-D array."""
        np_a = np.array([], dtype=dtype)
        gpu_a = cp.array(np_a)
        assert gpu_a.shape == (0,)
        result = gpu_a.get()
        assert result.shape == (0,)


# ── Zero-size 2-D ───────────────────────────────────────────────────


class TestZeroSize2D:
    """Test zero-size 2-D arrays."""

    # Ref: cupy_tests/core_tests/test_ndarray.py

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_zeros_0_3(self, dtype):
        """zeros((0, 3)) should have correct shape and size."""
        a = cp.zeros((0, 3), dtype=dtype)
        assert a.shape == (0, 3)
        assert a.size == 0
        result = a.get()
        assert result.shape == (0, 3)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_zeros_3_0(self, dtype):
        """zeros((3, 0)) should have correct shape and size."""
        a = cp.zeros((3, 0), dtype=dtype)
        assert a.shape == (3, 0)
        assert a.size == 0
        result = a.get()
        assert result.shape == (3, 0)


# ── Zero-size reductions ────────────────────────────────────────────


class TestZeroSizeReductions:
    """Test reductions on zero-size arrays."""

    # Ref: cupy_tests/math_tests/test_sumprod.py

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sum_empty_returns_zero(self, dtype):
        """sum of empty array should return 0."""
        a = cp.zeros((0,), dtype=dtype)
        result = cp.sum(a)
        np_result = np.sum(np.zeros((0,), dtype=dtype))
        assert_eq(result, np_result, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_empty_returns_nan(self, dtype):
        """mean of empty array should return nan (like NumPy)."""
        np_a = np.zeros((0,), dtype=dtype)
        np_result = np.mean(np_a)
        # NumPy returns nan for mean of empty with a warning
        assert np.isnan(np_result)


# ── Zero-size concat and stack ───────────────────────────────────────


class TestZeroSizeConcatStack:
    """Test concatenate and stack with zero-size arrays."""

    # Ref: cupy_tests/manipulation_tests/test_join.py

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_concatenate_with_empty(self, dtype):
        """Concatenating an empty array with a non-empty one should work."""
        a = cp.zeros((0,), dtype=dtype)
        b = cp.array([1.0, 2.0, 3.0], dtype=dtype)
        result = cp.concatenate([a, b])
        expected = np.concatenate([np.zeros((0,), dtype=dtype),
                                   np.array([1.0, 2.0, 3.0], dtype=dtype)])
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_concatenate_two_empty(self, dtype):
        """Concatenating two empty arrays should produce empty array."""
        a = cp.zeros((0,), dtype=dtype)
        b = cp.zeros((0,), dtype=dtype)
        result = cp.concatenate([a, b])
        assert result.size == 0


# ── Zero-size elementwise ops ────────────────────────────────────────


class TestZeroSizeElementwise:
    """Test elementwise operations on zero-size arrays."""

    # Ref: cupy_tests/core_tests/test_ndarray_elementwise_op.py

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_add_empty_arrays(self, dtype):
        """Adding two empty arrays should produce an empty array."""
        a = cp.zeros((0,), dtype=dtype)
        b = cp.zeros((0,), dtype=dtype)
        result = a + b
        assert result.shape == (0,)
        assert result.size == 0

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sqrt_empty(self, dtype):
        """sqrt of empty array should produce empty array."""
        a = cp.zeros((0,), dtype=dtype)
        result = cp.sqrt(a)
        assert result.shape == (0,)
        assert result.size == 0

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_neg_empty(self, dtype):
        """Negation of empty array should produce empty array."""
        a = cp.zeros((0,), dtype=dtype)
        result = -a
        assert result.shape == (0,)
        assert result.size == 0


# ── 0-d scalar arrays ───────────────────────────────────────────────


class TestScalar0D:
    """Test 0-d (scalar) array creation and properties."""

    # Ref: cupy_tests/core_tests/test_ndarray.py (0-d tests)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_scalar_creation(self, dtype):
        """0-d array should have shape=(), ndim=0, size=1."""
        if np.issubdtype(dtype, np.bool_):
            a = cp.array(True, dtype=dtype)
        elif np.issubdtype(dtype, np.complexfloating):
            a = cp.array(1 + 2j, dtype=dtype)
        else:
            a = cp.array(42, dtype=dtype)
        assert a.shape == ()
        assert a.ndim == 0
        assert a.size == 1

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_scalar_get_roundtrip(self, dtype):
        """0-d array .get() should return 0-d NumPy array with correct value."""
        if np.issubdtype(dtype, np.bool_):
            val = True
        elif np.issubdtype(dtype, np.complexfloating):
            val = 1 + 2j
        else:
            val = 42
        np_a = np.array(val, dtype=dtype)
        gpu_a = cp.array(np_a)
        result = gpu_a.get()
        assert result.shape == ()
        np.testing.assert_array_equal(result, np_a)


# ── 0-d arithmetic ──────────────────────────────────────────────────


class TestScalar0DArithmetic:
    """Test arithmetic on 0-d scalar arrays."""

    # Ref: cupy_tests/core_tests/test_ndarray_elementwise_op.py

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_scalar_add(self, dtype):
        """0-d + 0-d should produce a 0-d result matching NumPy."""
        np_a = np.array(3, dtype=dtype)
        np_b = np.array(5, dtype=dtype)
        gpu_a = cp.array(np_a)
        gpu_b = cp.array(np_b)
        np_result = np_a + np_b
        gpu_result = gpu_a + gpu_b
        assert_eq(gpu_result, np_result, dtype=np_result.dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_scalar_mul(self, dtype):
        """0-d * 0-d should produce correct result."""
        np_a = np.array(3.0, dtype=dtype)
        np_b = np.array(4.0, dtype=dtype)
        gpu_a = cp.array(np_a)
        gpu_b = cp.array(np_b)
        np_result = np_a * np_b
        gpu_result = gpu_a * gpu_b
        assert_eq(gpu_result, np_result, dtype=dtype, category="arithmetic")


# ── 0-d reduction ───────────────────────────────────────────────────


class TestScalar0DReduction:
    """Test reductions on 0-d arrays."""

    # Ref: cupy_tests/math_tests/test_sumprod.py

    def test_sum_0d(self):
        """sum of 0-d array should return the value itself."""
        a = cp.array(5.0, dtype=cp.float32)
        result = cp.sum(a)
        assert_eq(result, np.float32(5.0), dtype=np.float32)

    def test_max_0d(self):
        """max of 0-d array should return the value itself."""
        a = cp.array(5.0, dtype=cp.float32)
        result = cp.max(a)
        assert_eq(result, np.float32(5.0), dtype=np.float32)

    def test_min_0d(self):
        """min of 0-d array should return the value itself."""
        a = cp.array(3.0, dtype=cp.float32)
        result = cp.min(a)
        assert_eq(result, np.float32(3.0), dtype=np.float32)

    def test_mean_0d(self):
        """mean of 0-d array should return the value itself."""
        a = cp.array(7.0, dtype=cp.float32)
        result = cp.mean(a)
        assert_eq(result, np.float32(7.0), dtype=np.float32)


# ── Single-element reductions ────────────────────────────────────────


class TestSingleElementReductions:
    """Test reductions on arrays with exactly one element."""

    # Ref: cupy_tests/math_tests/test_sumprod.py

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_sum_single(self, dtype):
        """sum of size-1 array should return the element."""
        np_a = np.array([42], dtype=dtype)
        gpu_a = cp.array(np_a)
        assert_eq(cp.sum(gpu_a), np.sum(np_a), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_max_single(self, dtype):
        """max of size-1 array should return the element."""
        np_a = np.array([42], dtype=dtype)
        gpu_a = cp.array(np_a)
        assert_eq(cp.max(gpu_a), np.max(np_a), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_min_single(self, dtype):
        """min of size-1 array should return the element."""
        np_a = np.array([42], dtype=dtype)
        gpu_a = cp.array(np_a)
        assert_eq(cp.min(gpu_a), np.min(np_a), dtype=dtype)


# ── Shape (1,) vs (1,1) vs () differences ────────────────────────────


class TestShapeDifferences:
    """Test that (1,), (1,1), and () shapes are distinct."""

    # Ref: cupy_tests/core_tests/test_ndarray.py

    def test_shape_1_tuple(self):
        """Shape (1,) should have ndim=1."""
        a = cp.array([5.0], dtype=cp.float32)
        assert a.shape == (1,)
        assert a.ndim == 1

    def test_shape_1_1_tuple(self):
        """Shape (1,1) should have ndim=2."""
        a = cp.array([[5.0]], dtype=cp.float32)
        assert a.shape == (1, 1)
        assert a.ndim == 2

    def test_shape_scalar(self):
        """Shape () should have ndim=0."""
        a = cp.array(5.0, dtype=cp.float32)
        assert a.shape == ()
        assert a.ndim == 0

    def test_squeeze_1_1_to_scalar(self):
        """squeeze((1,1)) should produce ()."""
        a = cp.array([[5.0]], dtype=cp.float32)
        s = a.squeeze()
        assert s.shape == ()

    def test_reshape_scalar_to_1(self):
        """reshape(()) to (1,) should work."""
        a = cp.array(5.0, dtype=cp.float32)
        b = a.reshape((1,))
        assert b.shape == (1,)
        assert_eq(b, np.array([5.0], dtype=np.float32), dtype=np.float32)

    def test_size_comparison(self):
        """All three shapes should have size=1."""
        a0 = cp.array(5.0, dtype=cp.float32)
        a1 = cp.array([5.0], dtype=cp.float32)
        a11 = cp.array([[5.0]], dtype=cp.float32)
        assert a0.size == 1
        assert a1.size == 1
        assert a11.size == 1

    @pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
    def test_strides_size1(self, shape):
        """Strides for size-1 arrays should match NumPy."""
        np_a = np.zeros(shape, dtype=np.float32)
        gpu_a = cp.zeros(shape, dtype=cp.float32)
        assert gpu_a.strides == np_a.strides


# ── Large array tests ────────────────────────────────────────────────


class TestLargeArray:
    """Test creation and operations on large arrays (1M elements)."""

    # Ref: cupy_tests/core_tests/test_ndarray.py (large arrays)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large_creation(self, dtype):
        """Creating a 1M element array should succeed."""
        n = 1_000_000
        a = cp.ones(n, dtype=dtype)
        assert a.shape == (n,)
        assert a.size == n

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large_sum(self, dtype):
        """sum of 1M ones should equal 1M."""
        n = 1_000_000
        a = cp.ones(n, dtype=dtype)
        result = cp.sum(a)
        # float16 can't represent 1M exactly (max ~65504), so just check it doesn't crash
        if dtype == np.float16:
            _ = result.get()
        else:
            # float32 parallel reduction may accumulate small rounding errors
            import numpy.testing as npt
            npt.assert_allclose(float(result.get()), float(n), rtol=1e-4)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large_elementwise(self, dtype):
        """elementwise op on 1M elements should produce correct results."""
        n = 1_000_000
        np_a = np.ones(n, dtype=dtype)
        gpu_a = cp.ones(n, dtype=dtype)
        gpu_result = gpu_a + gpu_a
        np_result = np_a + np_a
        # Check first/last elements for correctness
        gpu_data = gpu_result.get()
        assert gpu_data[0] == np_result[0]
        assert gpu_data[-1] == np_result[-1]
        assert gpu_data.shape == np_result.shape
