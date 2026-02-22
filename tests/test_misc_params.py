"""Tests for param gaps in math_ops.py, creation.py, manipulation.py.

TDD: these tests are written first, then the source is updated to pass them.
All new params have defaults that preserve current behavior.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp


# ===================================================================
# math_ops.py — HIGH
# ===================================================================


class TestAroundOut:
    def test_around_out_basic(self):
        a = cp.array([1.567, 2.345, 3.789])
        out = cp.zeros(3)
        result = cp.around(a, decimals=1, out=out)
        expected = np.around(np.array([1.567, 2.345, 3.789], dtype=np.float32), decimals=1)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)
        npt.assert_allclose(out.get(), expected, rtol=1e-5)
        # result IS out
        assert result is out

    def test_around_no_out_unchanged(self):
        """Default out=None preserves current behavior."""
        a = cp.array([1.567, 2.345])
        result = cp.around(a, decimals=2)
        expected = np.around(np.array([1.567, 2.345], dtype=np.float32), decimals=2)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestRound_Out:
    def test_round__out_basic(self):
        from macmetalpy.math_ops import round_
        a = cp.array([1.567, 2.345, 3.789])
        out = cp.zeros(3)
        result = round_(a, decimals=1, out=out)
        expected = np.around(np.array([1.567, 2.345, 3.789], dtype=np.float32), decimals=1)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)
        npt.assert_allclose(out.get(), expected, rtol=1e-5)
        assert result is out


class TestClipOut:
    def test_clip_out_basic(self):
        a = cp.array([1.0, 5.0, 10.0])
        out = cp.zeros(3)
        result = cp.clip(a, 2.0, 8.0, out=out)
        expected = np.clip(np.array([1.0, 5.0, 10.0], dtype=np.float32), 2.0, 8.0)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)
        npt.assert_allclose(out.get(), expected, rtol=1e-5)
        assert result is out

    def test_clip_no_out_unchanged(self):
        a = cp.array([1.0, 5.0, 10.0])
        result = cp.clip(a, 2.0, 8.0)
        expected = np.clip(np.array([1.0, 5.0, 10.0], dtype=np.float32), 2.0, 8.0)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestTraceDtypeOut:
    def test_trace_dtype(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.trace(a, dtype=np.float32)
        expected = np.trace(np.array([[1, 2], [3, 4]], dtype=np.float32))
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_trace_out(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]])
        out = cp.zeros(())
        result = cp.trace(a, out=out)
        expected = np.trace(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        npt.assert_allclose(result.get(), expected, rtol=1e-5)
        npt.assert_allclose(out.get(), expected, rtol=1e-5)
        assert result is out

    def test_trace_no_extra_params_unchanged(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]])
        result = cp.trace(a)
        npt.assert_allclose(result.get(), np.float32(5.0), rtol=1e-5)


class TestNanToNumCopy:
    def test_nan_to_num_copy_true(self):
        a = cp.array([1.0, float('nan'), float('inf')])
        result = cp.nan_to_num(a, copy=True)
        # original unchanged
        assert np.isnan(a.get()[1])
        # result has no nan
        assert not np.isnan(result.get()[1])

    def test_nan_to_num_copy_false(self):
        a = cp.array([1.0, float('nan'), float('inf')])
        result = cp.nan_to_num(a, copy=False)
        # When copy=False, modifies in-place and returns same array
        assert result is a
        assert not np.isnan(a.get()[1])

    def test_nan_to_num_default_copy(self):
        """Default copy=True preserves current behavior (returns new array)."""
        a = cp.array([1.0, float('nan')])
        result = cp.nan_to_num(a)
        assert result is not a


class TestIsNegInfOut:
    def test_isneginf_out(self):
        a = cp.array([1.0, float('-inf'), float('inf'), 0.0])
        out = cp.zeros(4, dtype=np.float32)
        result = cp.isneginf(a, out=out)
        expected = np.isneginf(np.array([1.0, float('-inf'), float('inf'), 0.0]))
        npt.assert_array_equal(result.get().astype(bool), expected)
        assert result is out

    def test_isneginf_no_out(self):
        a = cp.array([1.0, float('-inf')])
        result = cp.isneginf(a)
        expected = np.isneginf(np.array([1.0, float('-inf')]))
        npt.assert_array_equal(result.get().astype(bool), expected)


class TestIsPosInfOut:
    def test_isposinf_out(self):
        a = cp.array([1.0, float('-inf'), float('inf'), 0.0])
        out = cp.zeros(4, dtype=np.float32)
        result = cp.isposinf(a, out=out)
        expected = np.isposinf(np.array([1.0, float('-inf'), float('inf'), 0.0]))
        npt.assert_array_equal(result.get().astype(bool), expected)
        assert result is out

    def test_isposinf_no_out(self):
        a = cp.array([1.0, float('inf')])
        result = cp.isposinf(a)
        expected = np.isposinf(np.array([1.0, float('inf')]))
        npt.assert_array_equal(result.get().astype(bool), expected)


# ===================================================================
# math_ops.py — MEDIUM
# ===================================================================


class TestCopyOrder:
    def test_copy_order_param_accepted(self):
        a = cp.array([1.0, 2.0, 3.0])
        result = cp.copy(a, order='C')
        npt.assert_allclose(result.get(), a.get())

    def test_copy_order_K_accepted(self):
        a = cp.array([1.0, 2.0, 3.0])
        result = cp.copy(a, order='K')
        npt.assert_allclose(result.get(), a.get())

    def test_copy_default_unchanged(self):
        a = cp.array([1.0, 2.0, 3.0])
        result = cp.copy(a)
        npt.assert_allclose(result.get(), a.get())


class TestAllcloseEqualNan:
    def test_allclose_equal_nan_true(self):
        a = cp.array([1.0, float('nan')])
        b = cp.array([1.0, float('nan')])
        assert cp.allclose(a, b, equal_nan=True)

    def test_allclose_equal_nan_false(self):
        a = cp.array([1.0, float('nan')])
        b = cp.array([1.0, float('nan')])
        assert not cp.allclose(a, b, equal_nan=False)

    def test_allclose_default_equal_nan(self):
        """Default equal_nan=False preserves current behavior."""
        a = cp.array([1.0, float('nan')])
        b = cp.array([1.0, float('nan')])
        assert not cp.allclose(a, b)


class TestIscloseEqualNan:
    def test_isclose_equal_nan_true(self):
        a = cp.array([1.0, float('nan')])
        b = cp.array([1.0, float('nan')])
        result = cp.isclose(a, b, equal_nan=True)
        expected = np.isclose(
            np.array([1.0, float('nan')]),
            np.array([1.0, float('nan')]),
            equal_nan=True,
        )
        npt.assert_array_equal(result.get(), expected)

    def test_isclose_equal_nan_false(self):
        a = cp.array([1.0, float('nan')])
        b = cp.array([1.0, float('nan')])
        result = cp.isclose(a, b, equal_nan=False)
        # NaN != NaN so second element is False
        assert result.get()[0] == True
        assert result.get()[1] == False

    def test_isclose_default(self):
        a = cp.array([1.0, 2.0])
        b = cp.array([1.0, 2.0])
        result = cp.isclose(a, b)
        npt.assert_array_equal(result.get(), [True, True])


class TestArrayEqualEqualNan:
    def test_array_equal_equal_nan_true(self):
        a = cp.array([1.0, float('nan')])
        b = cp.array([1.0, float('nan')])
        assert cp.array_equal(a, b, equal_nan=True)

    def test_array_equal_equal_nan_false(self):
        a = cp.array([1.0, float('nan')])
        b = cp.array([1.0, float('nan')])
        assert not cp.array_equal(a, b, equal_nan=False)

    def test_array_equal_default(self):
        """Default equal_nan=False: NaN != NaN."""
        a = cp.array([1.0, float('nan')])
        b = cp.array([1.0, float('nan')])
        assert not cp.array_equal(a, b)


# ===================================================================
# creation.py — HIGH
# ===================================================================


class TestGeomspaceEndpointAxis:
    def test_geomspace_endpoint_false(self):
        result = cp.geomspace(1, 1000, num=4, endpoint=False)
        expected = np.geomspace(1, 1000, num=4, endpoint=False).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_geomspace_endpoint_true_default(self):
        result = cp.geomspace(1, 1000, num=4)
        expected = np.geomspace(1, 1000, num=4).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_geomspace_axis(self):
        result = cp.geomspace(1, 1000, num=3, axis=0)
        expected = np.geomspace(1, 1000, num=3, axis=0).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestLinspaceAxis:
    def test_linspace_axis_0(self):
        result = cp.linspace([0, 5], [10, 15], num=3, axis=0)
        expected = np.linspace([0, 5], [10, 15], num=3, axis=0).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_linspace_axis_1(self):
        result = cp.linspace([0, 5], [10, 15], num=3, axis=1)
        expected = np.linspace([0, 5], [10, 15], num=3, axis=1).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_linspace_default_axis(self):
        """Default axis=0 preserves current behavior."""
        result = cp.linspace(0, 10, num=5)
        expected = np.linspace(0, 10, num=5).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestLogspaceBaseEndpointAxis:
    def test_logspace_base(self):
        result = cp.logspace(0, 2, num=3, base=2.0)
        expected = np.logspace(0, 2, num=3, base=2.0).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_logspace_endpoint_false(self):
        result = cp.logspace(0, 2, num=4, endpoint=False)
        expected = np.logspace(0, 2, num=4, endpoint=False).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_logspace_axis(self):
        result = cp.logspace(0, 2, num=3, axis=0)
        expected = np.logspace(0, 2, num=3, axis=0).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_logspace_default_unchanged(self):
        result = cp.logspace(0, 2, num=3)
        expected = np.logspace(0, 2, num=3).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestMeshgridCopySparse:
    def test_meshgrid_copy_false(self):
        x = cp.array([1, 2, 3])
        y = cp.array([4, 5])
        result = cp.meshgrid(x, y, copy=False)
        expected = np.meshgrid([1, 2, 3], [4, 5], copy=False)
        npt.assert_array_equal(result[0].get(), expected[0].astype(np.float32))
        npt.assert_array_equal(result[1].get(), expected[1].astype(np.float32))

    def test_meshgrid_sparse_true(self):
        x = cp.array([1.0, 2.0, 3.0])
        y = cp.array([4.0, 5.0])
        result = cp.meshgrid(x, y, sparse=True)
        expected = np.meshgrid([1.0, 2.0, 3.0], [4.0, 5.0], sparse=True)
        assert result[0].shape == expected[0].shape
        assert result[1].shape == expected[1].shape
        npt.assert_allclose(result[0].get(), expected[0].astype(np.float32), rtol=1e-5)
        npt.assert_allclose(result[1].get(), expected[1].astype(np.float32), rtol=1e-5)

    def test_meshgrid_default_unchanged(self):
        x = cp.array([1.0, 2.0])
        y = cp.array([3.0, 4.0])
        result = cp.meshgrid(x, y)
        expected = np.meshgrid([1.0, 2.0], [3.0, 4.0])
        npt.assert_allclose(result[0].get(), expected[0].astype(np.float32), rtol=1e-5)


# ===================================================================
# creation.py — MEDIUM (order params)
# ===================================================================


class TestEyeOrder:
    def test_eye_k_and_order(self):
        result = cp.eye(3, k=1, order='C')
        expected = np.eye(3, k=1).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_eye_default_unchanged(self):
        result = cp.eye(3)
        expected = np.eye(3).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestFullOrder:
    def test_full_order(self):
        result = cp.full((2, 3), 7.0, order='F')
        expected = np.full((2, 3), 7.0).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestFullLikeOrder:
    def test_full_like_order(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]])
        result = cp.full_like(a, 5.0, order='K')
        expected = np.full_like(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 5.0)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestOnesOrder:
    def test_ones_order(self):
        result = cp.ones((2, 3), order='F')
        expected = np.ones((2, 3)).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestOnesLikeOrder:
    def test_ones_like_order(self):
        a = cp.array([1.0, 2.0])
        result = cp.ones_like(a, order='K')
        expected = np.ones_like(np.array([1.0, 2.0], dtype=np.float32))
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestZerosLikeOrder:
    def test_zeros_like_order(self):
        a = cp.array([1.0, 2.0])
        result = cp.zeros_like(a, order='K')
        expected = np.zeros_like(np.array([1.0, 2.0], dtype=np.float32))
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


# ===================================================================
# manipulation.py — HIGH
# ===================================================================


class TestStackOutDtype:
    def test_stack_out(self):
        a = cp.array([1.0, 2.0])
        b = cp.array([3.0, 4.0])
        out = cp.zeros((2, 2))
        result = cp.stack([a, b], out=out)
        expected = np.stack([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
        npt.assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)
        npt.assert_allclose(out.get(), expected.astype(np.float32), rtol=1e-5)
        assert result is out

    def test_stack_dtype(self):
        a = cp.array([1, 2], dtype=np.int32)
        b = cp.array([3, 4], dtype=np.int32)
        result = cp.stack([a, b], dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), [[1.0, 2.0], [3.0, 4.0]], rtol=1e-5)

    def test_stack_default_unchanged(self):
        a = cp.array([1.0, 2.0])
        b = cp.array([3.0, 4.0])
        result = cp.stack([a, b])
        expected = np.stack([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
        npt.assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)


class TestHstackDtype:
    def test_hstack_dtype(self):
        a = cp.array([1, 2], dtype=np.int32)
        b = cp.array([3, 4], dtype=np.int32)
        result = cp.hstack([a, b], dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), [1.0, 2.0, 3.0, 4.0], rtol=1e-5)

    def test_hstack_default_unchanged(self):
        a = cp.array([1.0, 2.0])
        b = cp.array([3.0, 4.0])
        result = cp.hstack([a, b])
        expected = np.hstack([np.array([1.0, 2.0]), np.array([3.0, 4.0])]).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestVstackDtype:
    def test_vstack_dtype(self):
        a = cp.array([1, 2], dtype=np.int32)
        b = cp.array([3, 4], dtype=np.int32)
        result = cp.vstack([a, b], dtype=np.float32)
        assert result.dtype == np.float32
        npt.assert_allclose(result.get(), [[1.0, 2.0], [3.0, 4.0]], rtol=1e-5)

    def test_vstack_default_unchanged(self):
        a = cp.array([1.0, 2.0])
        b = cp.array([3.0, 4.0])
        result = cp.vstack([a, b])
        expected = np.vstack([np.array([1.0, 2.0]), np.array([3.0, 4.0])]).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


# ===================================================================
# manipulation.py — MEDIUM
# ===================================================================


class TestStackCasting:
    def test_stack_casting_param_accepted(self):
        a = cp.array([1.0, 2.0])
        b = cp.array([3.0, 4.0])
        result = cp.stack([a, b], casting='same_kind')
        expected = np.stack([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
        npt.assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)


class TestHstackCasting:
    def test_hstack_casting_param_accepted(self):
        a = cp.array([1.0, 2.0])
        b = cp.array([3.0, 4.0])
        result = cp.hstack([a, b], casting='same_kind')
        expected = np.hstack([np.array([1.0, 2.0]), np.array([3.0, 4.0])]).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestVstackCasting:
    def test_vstack_casting_param_accepted(self):
        a = cp.array([1.0, 2.0])
        b = cp.array([3.0, 4.0])
        result = cp.vstack([a, b], casting='same_kind')
        expected = np.vstack([np.array([1.0, 2.0]), np.array([3.0, 4.0])]).astype(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestReshapeOrder:
    def test_reshape_order(self):
        a = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = cp.reshape(a, (6,), order='C')
        expected = np.reshape(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), (6,), order='C')
        npt.assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)

    def test_reshape_default_unchanged(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]])
        result = cp.reshape(a, (4,))
        expected = np.reshape(np.array([[1.0, 2.0], [3.0, 4.0]]), (4,))
        npt.assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)


class TestRavelOrder:
    def test_ravel_order(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]])
        result = cp.ravel(a, order='C')
        expected = np.ravel(np.array([[1.0, 2.0], [3.0, 4.0]]), order='C')
        npt.assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)

    def test_ravel_default_unchanged(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]])
        result = cp.ravel(a)
        expected = np.ravel(np.array([[1.0, 2.0], [3.0, 4.0]]))
        npt.assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)
