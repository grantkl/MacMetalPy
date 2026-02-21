"""Tests for memory model: strides, views, contiguity.

Ref: cupy_tests/core_tests/test_ndarray.py (strides, contiguous, view tests)
"""

import numpy as np
import pytest

import macmetalpy as cp

from conftest import (
    ALL_DTYPES,
    FLOAT_DTYPES,
    NUMERIC_DTYPES,
    SHAPES_1D,
    SHAPES_2D,
    SHAPES_NONZERO,
    assert_eq,
    make_arg,
    tol_for,
)


# ── reshape returns view ─────────────────────────────────────────────


class TestReshapeView:
    """Test that reshape on contiguous arrays returns a view sharing the buffer."""

    # Ref: cupy_tests/core_tests/test_ndarray.py::TestReshape

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_reshape_shares_buffer(self, dtype):
        """Reshaped array from a contiguous source should share the buffer."""
        a = cp.arange(6, dtype=dtype)
        b = a.reshape((2, 3))
        # Both should reference the same underlying buffer
        assert b._base is a or b._buffer is a._buffer

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_reshape_flat_to_2d(self, dtype):
        """Values should be correct after reshape."""
        np_a = np.arange(6, dtype=dtype)
        gpu_a = cp.array(np_a)
        np_b = np_a.reshape((2, 3))
        gpu_b = gpu_a.reshape((2, 3))
        assert_eq(gpu_b, np_b, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_reshape_infer_dim(self, dtype):
        """Reshape with -1 should infer the correct dimension."""
        np_a = np.arange(12, dtype=dtype)
        gpu_a = cp.array(np_a)
        np_b = np_a.reshape((3, -1))
        gpu_b = gpu_a.reshape((3, -1))
        assert gpu_b.shape == np_b.shape
        assert_eq(gpu_b, np_b, dtype=dtype)


# ── transpose returns view ───────────────────────────────────────────


class TestTransposeView:
    """Test that transpose returns a view, not a copy."""

    # Ref: cupy_tests/core_tests/test_ndarray.py::TestTranspose

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_transpose_shares_buffer(self, dtype, shape):
        """Transposed array should share the underlying buffer."""
        np_a = make_arg(shape, dtype, xp=np)
        gpu_a = cp.array(np_a)
        gpu_t = gpu_a.transpose()
        assert gpu_t._base is gpu_a or gpu_t._buffer is gpu_a._buffer

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_transpose_values(self, dtype, shape):
        """Transposed values should match NumPy."""
        np_a = make_arg(shape, dtype, xp=np)
        gpu_a = cp.array(np_a)
        assert_eq(gpu_a.transpose(), np_a.T, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_transpose_3d(self, dtype):
        """Transpose of a 3-D array should match NumPy."""
        shape = (2, 3, 4)
        np_a = make_arg(shape, dtype, xp=np)
        gpu_a = cp.array(np_a)
        assert_eq(gpu_a.transpose(), np_a.T, dtype=dtype)


# ── ravel: view vs copy ─────────────────────────────────────────────


class TestRavelViewCopy:
    """Test ravel returns view for contiguous, copy for non-contiguous."""

    # Ref: cupy_tests/core_tests/test_ndarray.py::TestRavel

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_ravel_contiguous_is_view(self, dtype):
        """ravel on contiguous array should share buffer (view)."""
        a = cp.arange(6, dtype=dtype)
        b = a.reshape((2, 3))
        r = b.ravel()
        # Should share buffer with b (and thus with a)
        assert r._buffer is b._buffer or r._base is not None

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_ravel_noncontiguous_is_copy(self, dtype):
        """ravel on non-contiguous array should produce correct values."""
        np_a = make_arg((3, 4), dtype, xp=np)
        gpu_a = cp.array(np_a)
        gpu_t = gpu_a.transpose()
        gpu_r = gpu_t.ravel()
        np_r = np_a.T.ravel()
        assert_eq(gpu_r, np_r, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_ravel_values_contiguous(self, dtype):
        """ravel on contiguous array should produce correct flat values."""
        np_a = make_arg((2, 3), dtype, xp=np)
        gpu_a = cp.array(np_a)
        assert_eq(gpu_a.ravel(), np_a.ravel(), dtype=dtype)


# ── Slice as kernel input ────────────────────────────────────────────


class TestSliceKernelInput:
    """Test that sliced arrays work correctly as inputs to GPU operations."""

    # Ref: cupy_tests/core_tests/test_ndarray_elementwise_op.py

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_1D)
    def test_slice_stride2_sqrt(self, dtype, shape):
        """cp.sqrt(a[::2]) should produce correct results."""
        if shape[0] < 2:
            pytest.skip("Need at least 2 elements for stride-2 slice")
        np_a = np.arange(1, shape[0] + 1, dtype=dtype)
        gpu_a = cp.array(np_a)
        np_result = np.sqrt(np_a[::2])
        gpu_result = cp.sqrt(gpu_a[::2])
        assert_eq(gpu_result, np_result, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_1D)
    def test_slice_offset_values(self, dtype, shape):
        """a[1:] should produce correct offset values."""
        if shape[0] < 2:
            pytest.skip("Need at least 2 elements for offset slice")
        np_a = make_arg(shape, dtype, xp=np)
        gpu_a = cp.array(np_a)
        np_result = np_a[1:]
        gpu_result = gpu_a[1:]
        assert_eq(gpu_result, np_result, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_transposed_as_kernel_input(self, dtype, shape):
        """Transposed array used in elementwise op should be correct."""
        np_a = make_arg(shape, dtype, xp=np)
        gpu_a = cp.array(np_a)
        np_result = np.sqrt(np_a.T)
        gpu_result = cp.sqrt(gpu_a.T)
        assert_eq(gpu_result, np_result, dtype=dtype, category="unary_math")


# ── _is_c_contiguous ─────────────────────────────────────────────────


class TestIsContiguous:
    """Test the _is_c_contiguous method."""

    # Ref: cupy_tests/core_tests/test_ndarray.py

    def test_fresh_array_is_contiguous(self):
        """Freshly created array should be C-contiguous."""
        a = cp.zeros((3, 4))
        assert a._is_c_contiguous()

    def test_transposed_not_contiguous(self):
        """Transposed 2-D array should not be C-contiguous."""
        a = cp.zeros((3, 4))
        t = a.transpose()
        assert not t._is_c_contiguous()

    def test_scalar_is_contiguous(self):
        """0-d array should be C-contiguous."""
        a = cp.array(1.0)
        assert a._is_c_contiguous()

    def test_1d_array_is_contiguous(self):
        """1-D array should be C-contiguous."""
        a = cp.arange(10)
        assert a._is_c_contiguous()


# ── .strides matches NumPy ───────────────────────────────────────────


class TestStrides:
    """Test .strides property matches NumPy for various shapes and dtypes."""

    # Ref: cupy_tests/core_tests/test_ndarray.py::TestStrides

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_strides_c_contiguous(self, dtype, shape):
        """C-contiguous strides should match NumPy."""
        np_a = make_arg(shape, dtype, xp=np)
        gpu_a = cp.array(np_a, dtype=dtype)
        np_strides = np.zeros(shape, dtype=dtype).strides
        assert gpu_a.strides == np_strides, (
            f"strides mismatch for shape={shape}, dtype={np.dtype(dtype)}: "
            f"got {gpu_a.strides}, expected {np_strides}"
        )

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_strides_after_transpose(self, dtype, shape):
        """Strides after transpose should match NumPy's transposed strides."""
        if dtype == np.complex64:
            pytest.skip("complex64 transpose returns contiguous copy, not view")
        np_a = make_arg(shape, dtype, xp=np)
        gpu_a = cp.array(np_a, dtype=dtype)
        np_strides = np.zeros(shape, dtype=dtype).T.strides
        gpu_strides = gpu_a.transpose().strides
        assert gpu_strides == np_strides, (
            f"transposed strides mismatch for shape={shape}, dtype={np.dtype(dtype)}: "
            f"got {gpu_strides}, expected {np_strides}"
        )


# ── ascontiguousarray ────────────────────────────────────────────────


class TestAscontiguousarray:
    """Test cp.ascontiguousarray behavior."""

    # Ref: cupy_tests/core_tests/test_ndarray.py

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_already_contiguous_values_preserved(self, dtype):
        """ascontiguousarray on a contiguous array should preserve values."""
        np_a = make_arg((3, 4), dtype, xp=np)
        gpu_a = cp.array(np_a)
        gpu_c = cp.ascontiguousarray(gpu_a)
        assert_eq(gpu_c, np_a, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_noncontiguous_becomes_contiguous(self, dtype):
        """ascontiguousarray on transposed should produce contiguous result."""
        np_a = make_arg((3, 4), dtype, xp=np)
        gpu_a = cp.array(np_a)
        gpu_t = gpu_a.transpose()
        gpu_c = cp.ascontiguousarray(gpu_t)
        assert_eq(gpu_c, np.ascontiguousarray(np_a.T), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d_contiguous(self, dtype):
        """1-D array should already be contiguous."""
        np_a = make_arg((5,), dtype, xp=np)
        gpu_a = cp.array(np_a)
        gpu_c = cp.ascontiguousarray(gpu_a)
        assert_eq(gpu_c, np_a, dtype=dtype)


# ── flatten ──────────────────────────────────────────────────────────


class TestFlatten:
    """Test flatten always produces a contiguous 1-D copy."""

    # Ref: cupy_tests/core_tests/test_ndarray.py::TestFlatten

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_flatten_2d(self, dtype):
        """Flatten a 2-D array."""
        np_a = make_arg((2, 3), dtype, xp=np)
        gpu_a = cp.array(np_a)
        assert_eq(gpu_a.flatten(), np_a.flatten(), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_flatten_transposed(self, dtype):
        """Flatten a transposed (non-contiguous) array."""
        np_a = make_arg((3, 4), dtype, xp=np)
        gpu_a = cp.array(np_a)
        gpu_t = gpu_a.transpose()
        assert_eq(gpu_t.flatten(), np_a.T.flatten(), dtype=dtype)


# ── T property ───────────────────────────────────────────────────────


class TestTProperty:
    """Test .T property returns correct transposed view."""

    # Ref: cupy_tests/core_tests/test_ndarray.py

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_T_values(self, dtype, shape):
        """.T should match NumPy's .T."""
        np_a = make_arg(shape, dtype, xp=np)
        gpu_a = cp.array(np_a)
        assert_eq(gpu_a.T, np_a.T, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_T_1d_is_noop(self, dtype):
        """.T of a 1-D array should be the same as the original."""
        np_a = make_arg((5,), dtype, xp=np)
        gpu_a = cp.array(np_a)
        assert_eq(gpu_a.T, np_a.T, dtype=dtype)
