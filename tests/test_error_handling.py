"""Tests for error paths: bad inputs, shape mismatches, invalid operations.

Ref: cupy_tests/core_tests/test_ndarray.py (error cases)
     cupy_tests/linalg_tests/ (linalg error cases)
"""

import numpy as np
import pytest

import macmetalpy as cp
from macmetalpy.raw_kernel import RawKernel


# ── Creation errors ──────────────────────────────────────────────────


class TestCreationErrors:
    """Test error handling in array creation functions."""

    # Ref: cupy_tests/creation_tests/test_basic.py

    def test_arange_step_zero(self):
        """arange with step=0 should raise."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            cp.arange(0, 10, 0)

    def test_array_int8_upcast(self):
        """Creating array with int8 should upcast to int16."""
        a = cp.array([1, 2, 3], dtype=np.int8)
        assert a.dtype == np.int16

    def test_array_uint8_upcast(self):
        """Creating array with uint8 should upcast to uint16."""
        a = cp.array([1, 2, 3], dtype=np.uint8)
        assert a.dtype == np.uint16


# ── Reshape errors ───────────────────────────────────────────────────


class TestReshapeErrors:
    """Test error handling in reshape."""

    # Ref: cupy_tests/core_tests/test_ndarray.py::TestReshape

    def test_reshape_incompatible_size(self):
        """Reshape to incompatible size should raise ValueError."""
        a = cp.zeros((2, 3))
        with pytest.raises(ValueError, match="cannot reshape"):
            a.reshape((4, 4))

    def test_reshape_two_negative_ones(self):
        """Reshape with two -1 dims should raise ValueError."""
        a = cp.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="can only specify one unknown dimension"):
            a.reshape((-1, -1))

    def test_reshape_incompatible_with_neg1(self):
        """Reshape with -1 and incompatible known dims should raise."""
        a = cp.zeros((2, 3))  # size = 6
        with pytest.raises(ValueError, match="cannot reshape"):
            a.reshape((4, -1))  # 4 does not divide 6 evenly

    def test_reshape_zero_size_incompatible(self):
        """Reshape zero-size array to incompatible shape should raise."""
        a = cp.zeros((0,))
        with pytest.raises(ValueError, match="cannot reshape"):
            a.reshape((2, 3))


# ── Squeeze errors ───────────────────────────────────────────────────


class TestSqueezeErrors:
    """Test error handling in squeeze."""

    # Ref: cupy_tests/core_tests/test_ndarray.py::TestSqueeze

    def test_squeeze_non_size1_axis(self):
        """Squeeze on axis with size != 1 should raise ValueError."""
        a = cp.zeros((2, 3))
        with pytest.raises(ValueError, match="cannot s"):
            a.squeeze(axis=0)

    def test_squeeze_non_size1_axis_negative(self):
        """Squeeze on negative axis with size != 1 should raise ValueError."""
        a = cp.zeros((2, 3))
        with pytest.raises(ValueError, match="cannot s"):
            a.squeeze(axis=-1)


# ── Matmul errors ────────────────────────────────────────────────────


class TestMatmulErrors:
    """Test error handling in matmul."""

    # Ref: cupy_tests/core_tests/test_ndarray.py::TestMatmul

    def test_matmul_shape_mismatch(self):
        """Matmul with incompatible inner dimensions should raise."""
        a = cp.zeros((2, 3), dtype=cp.float32)
        b = cp.zeros((4, 5), dtype=cp.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            a @ b

    def test_matmul_1d_inner_product(self):
        """Matmul with 1-D arrays computes inner product (like NumPy)."""
        import numpy as np
        a = cp.arange(3, dtype=cp.float32)
        b = cp.arange(3, dtype=cp.float32)
        result = a @ b
        expected = np.arange(3, dtype=np.float32) @ np.arange(3, dtype=np.float32)
        assert float(result.get()) == float(expected)

    def test_matmul_3d_raises(self):
        """Matmul with 3-D array should raise."""
        a = cp.zeros((2, 3, 4), dtype=cp.float32)
        b = cp.zeros((4, 5), dtype=cp.float32)
        with pytest.raises(ValueError):
            a @ b


# ── Linalg errors ────────────────────────────────────────────────────


class TestLinalgErrors:
    """Test error handling in linalg functions."""

    # Ref: cupy_tests/linalg_tests/test_solve.py, test_decomposition.py

    def test_inv_non_square(self):
        """inv of non-square matrix should raise."""
        a = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cp.float32)
        with pytest.raises(np.linalg.LinAlgError):
            cp.linalg.inv(a)

    def test_inv_singular(self):
        """inv of singular matrix should raise."""
        a = cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.float32)
        with pytest.raises(np.linalg.LinAlgError):
            cp.linalg.inv(a)

    def test_det_non_square(self):
        """det of non-square matrix should raise."""
        a = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cp.float32)
        with pytest.raises(np.linalg.LinAlgError):
            cp.linalg.det(a)

    def test_solve_singular(self):
        """solve with singular matrix should raise."""
        a = cp.array([[0.0, 0.0], [0.0, 0.0]], dtype=cp.float32)
        b = cp.array([1.0, 2.0], dtype=cp.float32)
        with pytest.raises(np.linalg.LinAlgError):
            cp.linalg.solve(a, b)

    def test_cholesky_non_positive_definite(self):
        """cholesky of non-positive-definite matrix should raise."""
        a = cp.array([[-1.0, 0.0], [0.0, -1.0]], dtype=cp.float32)
        with pytest.raises(np.linalg.LinAlgError):
            cp.linalg.cholesky(a)

    def test_cholesky_non_square(self):
        """cholesky of non-square matrix should raise."""
        a = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cp.float32)
        with pytest.raises(np.linalg.LinAlgError):
            cp.linalg.cholesky(a)


# ── Broadcast errors ─────────────────────────────────────────────────


class TestBroadcastErrors:
    """Test error handling for incompatible broadcast shapes."""

    # Ref: cupy_tests/core_tests/test_ndarray_elementwise_op.py

    def test_incompatible_shapes_add(self):
        """Adding arrays with incompatible shapes should raise."""
        a = cp.zeros((2, 3), dtype=cp.float32)
        b = cp.zeros((4, 5), dtype=cp.float32)
        with pytest.raises(ValueError):
            a + b

    def test_incompatible_shapes_mul(self):
        """Multiplying arrays with incompatible shapes should raise."""
        a = cp.zeros((2, 3), dtype=cp.float32)
        b = cp.zeros((2, 4), dtype=cp.float32)
        with pytest.raises(ValueError):
            a * b

    def test_broadcast_to_incompatible(self):
        """broadcast_to with incompatible target shape should raise."""
        a = cp.zeros((3,), dtype=cp.float32)
        with pytest.raises(ValueError):
            cp.broadcast_to(a, (4,))


# ── Concat/stack errors ──────────────────────────────────────────────


class TestConcatStackErrors:
    """Test error handling in concatenate and stack."""

    # Ref: cupy_tests/manipulation_tests/test_join.py

    def test_concatenate_shape_mismatch(self):
        """Concatenating arrays with mismatched non-concat dims should raise."""
        a = cp.zeros((2, 3), dtype=cp.float32)
        b = cp.zeros((2, 4), dtype=cp.float32)
        with pytest.raises(ValueError):
            cp.concatenate([a, b], axis=0)

    def test_stack_shape_mismatch(self):
        """Stacking arrays with different shapes should raise."""
        a = cp.zeros((2, 3), dtype=cp.float32)
        b = cp.zeros((3, 4), dtype=cp.float32)
        with pytest.raises(ValueError):
            cp.stack([a, b])

    def test_concatenate_empty_list(self):
        """Concatenating empty list should raise."""
        with pytest.raises(ValueError):
            cp.concatenate([])


# ── Index errors ─────────────────────────────────────────────────────


class TestIndexErrors:
    """Test error handling in indexing."""

    # Ref: cupy_tests/indexing_tests/test_indexing.py

    def test_getitem_out_of_bounds_positive(self):
        """Positive out-of-bounds index should raise IndexError."""
        a = cp.arange(5, dtype=cp.float32)
        with pytest.raises(IndexError):
            _ = a[10]

    def test_getitem_out_of_bounds_negative(self):
        """Negative out-of-bounds index should raise IndexError."""
        a = cp.arange(5, dtype=cp.float32)
        with pytest.raises(IndexError):
            _ = a[-10]


# ── Scalar conversion errors ─────────────────────────────────────────


class TestScalarConversionErrors:
    """Test error handling for float()/int()/len() on arrays."""

    # Ref: cupy_tests/core_tests/test_ndarray.py

    def test_float_on_multi_element(self):
        """float() on multi-element array should raise TypeError."""
        a = cp.arange(5, dtype=cp.float32)
        with pytest.raises(TypeError, match="size-1"):
            float(a)

    def test_int_on_multi_element(self):
        """int() on multi-element array should raise TypeError."""
        a = cp.arange(5, dtype=cp.int32)
        with pytest.raises(TypeError, match="size-1"):
            int(a)

    def test_len_on_0d(self):
        """len() on 0-d array should raise TypeError."""
        a = cp.array(1.0)
        with pytest.raises(TypeError, match="unsized"):
            len(a)

    def test_float_on_size1_ok(self):
        """float() on size-1 array should work."""
        a = cp.array([3.14], dtype=cp.float32)
        result = float(a)
        assert isinstance(result, float)

    def test_int_on_size1_ok(self):
        """int() on size-1 array should work."""
        a = cp.array([42], dtype=cp.int32)
        result = int(a)
        assert result == 42


# ── Split errors ─────────────────────────────────────────────────────


class TestSplitErrors:
    """Test error handling in split."""

    # Ref: cupy_tests/manipulation_tests/test_split.py

    def test_split_uneven(self):
        """split with sections that don't divide evenly should raise."""
        a = cp.arange(7, dtype=cp.float32)
        with pytest.raises(ValueError):
            cp.split(a, 3)

    def test_split_too_many_sections(self):
        """split with more sections than elements should raise."""
        a = cp.arange(3, dtype=cp.float32)
        with pytest.raises(ValueError):
            cp.split(a, 5)


# ── Invalid axis errors ─────────────────────────────────────────────


class TestInvalidAxisErrors:
    """Test error handling for invalid axis arguments."""

    # Ref: cupy_tests/core_tests/test_ndarray.py

    def test_sum_invalid_axis(self):
        """sum with axis out of range should raise."""
        a = cp.zeros((2, 3), dtype=cp.float32)
        with pytest.raises((np.exceptions.AxisError, IndexError, ValueError)):
            cp.sum(a, axis=5)

    def test_concatenate_invalid_axis(self):
        """concatenate with invalid axis should raise."""
        a = cp.zeros((2, 3), dtype=cp.float32)
        b = cp.zeros((2, 3), dtype=cp.float32)
        with pytest.raises((np.exceptions.AxisError, IndexError, ValueError)):
            cp.concatenate([a, b], axis=5)


# ── RawKernel errors ─────────────────────────────────────────────────


class TestRawKernelErrors:
    """Test error handling in RawKernel."""

    # Ref: cupy_tests/core_tests/test_raw.py

    def test_bad_grid_tuple_4_elements(self):
        """Grid tuple with >3 elements should raise ValueError."""
        source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void dummy(device float* out [[buffer(0)]],
                          uint id [[thread_position_in_grid]]) {
            out[id] = 1.0;
        }
        """
        kernel = RawKernel(source, "dummy")
        out = cp.zeros(4, dtype=cp.float32)
        with pytest.raises(ValueError, match="1-3 elements"):
            kernel((1, 1, 1, 1), [out])

    def test_bad_grid_tuple_5_elements(self):
        """Grid tuple with 5 elements should raise ValueError."""
        source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void dummy(device float* out [[buffer(0)]],
                          uint id [[thread_position_in_grid]]) {
            out[id] = 1.0;
        }
        """
        kernel = RawKernel(source, "dummy")
        out = cp.zeros(4, dtype=cp.float32)
        with pytest.raises(ValueError, match="1-3 elements"):
            kernel((1, 1, 1, 1, 1), [out])
