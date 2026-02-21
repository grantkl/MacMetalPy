"""Tests for sorting enhancements: unique return_index/inverse/counts, axis, equal_nan.

Covers:
- unique with return_index
- unique with return_inverse
- unique with return_counts
- unique with combinations of return options
- unique with axis
- unique with equal_nan
- Backward compatibility: unique(a) still works
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp

from conftest import NUMERIC_DTYPES, FLOAT_DTYPES, tol_for


# ======================================================================
# unique with return_index
# ======================================================================

class TestUniqueReturnIndex:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_arr = np.array([3, 1, 2, 1, 3, 2], dtype=dtype)
        result, indices = cp.unique(cp.array(np_arr), return_index=True)
        expected, exp_idx = np.unique(np_arr, return_index=True)
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(indices.get(), exp_idx)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_same(self, dtype):
        np_arr = np.array([5, 5, 5], dtype=dtype)
        result, indices = cp.unique(cp.array(np_arr), return_index=True)
        expected, exp_idx = np.unique(np_arr, return_index=True)
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(indices.get(), exp_idx)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_already_sorted(self, dtype):
        np_arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result, indices = cp.unique(cp.array(np_arr), return_index=True)
        expected, exp_idx = np.unique(np_arr, return_index=True)
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(indices.get(), exp_idx)


# ======================================================================
# unique with return_inverse
# ======================================================================

class TestUniqueReturnInverse:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_arr = np.array([3, 1, 2, 1, 3, 2], dtype=dtype)
        result, inverse = cp.unique(cp.array(np_arr), return_inverse=True)
        expected, exp_inv = np.unique(np_arr, return_inverse=True)
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(inverse.get(), exp_inv)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_reconstruction(self, dtype):
        """Verify input can be reconstructed from unique and inverse."""
        np_arr = np.array([3, 1, 2, 1, 3, 2], dtype=dtype)
        result, inverse = cp.unique(cp.array(np_arr), return_inverse=True)
        # Reconstruct: result[inverse] should equal original
        reconstructed = result.get()[inverse.get()]
        npt.assert_array_equal(reconstructed, np_arr)


# ======================================================================
# unique with return_counts
# ======================================================================

class TestUniqueReturnCounts:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_arr = np.array([3, 1, 2, 1, 3, 2], dtype=dtype)
        result, counts = cp.unique(cp.array(np_arr), return_counts=True)
        expected, exp_counts = np.unique(np_arr, return_counts=True)
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(counts.get(), exp_counts)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_same(self, dtype):
        np_arr = np.array([5, 5, 5, 5], dtype=dtype)
        result, counts = cp.unique(cp.array(np_arr), return_counts=True)
        expected, exp_counts = np.unique(np_arr, return_counts=True)
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(counts.get(), exp_counts)


# ======================================================================
# unique with multiple return options
# ======================================================================

class TestUniqueMultiReturn:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_index_and_inverse(self, dtype):
        np_arr = np.array([3, 1, 2, 1, 3], dtype=dtype)
        result, idx, inv = cp.unique(
            cp.array(np_arr), return_index=True, return_inverse=True
        )
        expected, exp_idx, exp_inv = np.unique(
            np_arr, return_index=True, return_inverse=True
        )
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(idx.get(), exp_idx)
        npt.assert_array_equal(inv.get(), exp_inv)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_index_and_counts(self, dtype):
        np_arr = np.array([3, 1, 2, 1, 3], dtype=dtype)
        result, idx, counts = cp.unique(
            cp.array(np_arr), return_index=True, return_counts=True
        )
        expected, exp_idx, exp_counts = np.unique(
            np_arr, return_index=True, return_counts=True
        )
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(idx.get(), exp_idx)
        npt.assert_array_equal(counts.get(), exp_counts)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_three(self, dtype):
        np_arr = np.array([3, 1, 2, 1, 3, 2], dtype=dtype)
        result, idx, inv, counts = cp.unique(
            cp.array(np_arr),
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        expected, exp_idx, exp_inv, exp_counts = np.unique(
            np_arr,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(idx.get(), exp_idx)
        npt.assert_array_equal(inv.get(), exp_inv)
        npt.assert_array_equal(counts.get(), exp_counts)


# ======================================================================
# unique with axis
# ======================================================================

class TestUniqueAxis:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis0_2d(self, dtype):
        np_arr = np.array([[1, 2], [3, 4], [1, 2]], dtype=dtype)
        result = cp.unique(cp.array(np_arr), axis=0)
        expected = np.unique(np_arr, axis=0)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis1_2d(self, dtype):
        np_arr = np.array([[1, 2, 1], [3, 4, 3]], dtype=dtype)
        result = cp.unique(cp.array(np_arr), axis=1)
        expected = np.unique(np_arr, axis=1)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis0_with_counts(self, dtype):
        np_arr = np.array([[1, 2], [3, 4], [1, 2]], dtype=dtype)
        result, counts = cp.unique(cp.array(np_arr), axis=0, return_counts=True)
        expected, exp_counts = np.unique(np_arr, axis=0, return_counts=True)
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(counts.get(), exp_counts)


# ======================================================================
# unique with equal_nan
# ======================================================================

class TestUniqueEqualNan:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_equal_nan_true(self, dtype):
        """With equal_nan=True (default), NaN values are considered equal."""
        np_arr = np.array([1, np.nan, 2, np.nan, 3], dtype=dtype)
        result = cp.unique(cp.array(np_arr), equal_nan=True)
        expected = np.unique(np_arr, equal_nan=True)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_equal_nan_false(self, dtype):
        """With equal_nan=False, each NaN is treated as unique."""
        np_arr = np.array([1, np.nan, 2, np.nan, 3], dtype=dtype)
        result = cp.unique(cp.array(np_arr), equal_nan=False)
        expected = np.unique(np_arr, equal_nan=False)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_equal_nan_with_counts(self, dtype):
        np_arr = np.array([1, np.nan, 2, np.nan, 3], dtype=dtype)
        result, counts = cp.unique(
            cp.array(np_arr), return_counts=True, equal_nan=True
        )
        expected, exp_counts = np.unique(
            np_arr, return_counts=True, equal_nan=True
        )
        npt.assert_array_equal(result.get(), expected)
        npt.assert_array_equal(counts.get(), exp_counts)


# ======================================================================
# Backward compatibility
# ======================================================================

class TestUniqueBackwardCompat:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_single_arg_still_works(self, dtype):
        """unique(a) should still return just the unique array, not a tuple."""
        np_arr = np.array([3, 1, 2, 1, 3], dtype=dtype)
        result = cp.unique(cp.array(np_arr))
        expected = np.unique(np_arr)
        # Should not be a tuple
        assert not isinstance(result, tuple)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_empty(self, dtype):
        np_arr = np.array([], dtype=dtype)
        result = cp.unique(cp.array(np_arr))
        assert result.get().shape == (0,)
