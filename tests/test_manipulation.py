"""Tests for array manipulation functions -- CuPy-parity parametrized suite.

Ref: cupy_tests/manipulation_tests/
~2,058 parametrized cases covering: reshape, transpose, ravel, squeeze,
tile, repeat, flip, fliplr, flipud, rot90, roll, split, array_split,
hsplit, vsplit, dsplit, concatenate, stack, vstack, hstack, dstack,
column_stack, concat, moveaxis, swapaxes, rollaxis, broadcast_to,
broadcast_arrays, atleast_1d/2d/3d, delete, append, resize, trim_zeros,
copyto, pad.
"""

import numpy as np
import pytest

import macmetalpy as cp
from conftest import (
    ALL_DTYPES, NUMERIC_DTYPES, FLOAT_DTYPES, BROADCAST_PAIRS,
    SHAPES_2D,
    assert_eq, make_arg, tol_for,
)


# ── Shape groups for manipulation ─────────────────────────────────────
MANIP_SHAPES = [(5,), (2, 3), (2, 3, 4)]


# ====================================================================
# Shape: reshape, transpose, ravel, flatten, squeeze
# ====================================================================
# Ref: cupy_tests/manipulation_tests/test_shape.py

class TestReshape:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_1d_to_2d(self, dtype):
        a_np = make_arg((6,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.reshape(a_cp, (2, 3)), np.reshape(a_np, (2, 3)), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_2d_to_1d(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.reshape(a_cp, (6,)), np.reshape(a_np, (6,)), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_infer_dim(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.reshape(a_cp, (-1, 4)), np.reshape(a_np, (-1, 4)), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_3d_reshape(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.reshape(a_cp, (6, 4)), np.reshape(a_np, (6, 4)), dtype=dtype)


class TestTranspose:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_default_2d(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.transpose(a_cp), np.transpose(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_axes_3d(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        result = cp.transpose(a_cp, axes=(1, 2, 0))
        expected = np.transpose(a_np, axes=(1, 2, 0))
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_reverse_axes(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        result = cp.transpose(a_cp, axes=(2, 1, 0))
        expected = np.transpose(a_np, axes=(2, 1, 0))
        assert_eq(result, expected, dtype=dtype)


class TestRavel:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", MANIP_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.ravel(a_cp)
        expected = np.ravel(a_np)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)


class TestSqueeze:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_all_dims(self, dtype):
        a_np = make_arg((1, 3, 1), dtype)
        a_cp = cp.array(a_np)
        result = cp.squeeze(a_cp)
        expected = np.squeeze(a_np)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_specific_axis(self, dtype):
        a_np = make_arg((1, 3, 1), dtype)
        a_cp = cp.array(a_np)
        result = cp.squeeze(a_cp, axis=0)
        expected = np.squeeze(a_np, axis=0)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_no_squeeze_needed(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        result = cp.squeeze(a_cp)
        assert result.shape == a_np.shape


# ====================================================================
# Tile / repeat
# ====================================================================
# Ref: cupy_tests/manipulation_tests/test_tiling.py

class TestTile:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_scalar_reps_1d(self, dtype):
        a_np = make_arg((3,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.tile(a_cp, 3), np.tile(a_np, 3), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_tuple_reps_2d(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.tile(a_cp, (2, 2)), np.tile(a_np, (2, 2)), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d_to_2d_reps(self, dtype):
        a_np = make_arg((3,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.tile(a_cp, (2, 1)), np.tile(a_np, (2, 1)), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_3d(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.tile(a_cp, 2), np.tile(a_np, 2), dtype=dtype)


class TestRepeat:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_flat(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.repeat(a_cp, 2), np.repeat(a_np, 2), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis0(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.repeat(a_cp, 3, axis=0), np.repeat(a_np, 3, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis1(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.repeat(a_cp, 2, axis=1), np.repeat(a_np, 2, axis=1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_no_axis_2d(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.repeat(a_cp, 2), np.repeat(a_np, 2), dtype=dtype)


# ====================================================================
# Flip (flip, fliplr, flipud, rot90)
# ====================================================================
# Ref: cupy_tests/manipulation_tests/test_flip.py

class TestFlip:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.flip(a_cp), np.flip(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_2d_no_axis(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.flip(a_cp), np.ascontiguousarray(np.flip(a_np)), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_2d_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.flip(a_cp, axis=0),
                  np.ascontiguousarray(np.flip(a_np, axis=0)), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_2d_axis1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.flip(a_cp, axis=1),
                  np.ascontiguousarray(np.flip(a_np, axis=1)), dtype=dtype)


class TestFliplr:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.fliplr(a_cp),
                  np.ascontiguousarray(np.fliplr(a_np)), dtype=dtype)


class TestFlipud:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.flipud(a_cp),
                  np.ascontiguousarray(np.flipud(a_np)), dtype=dtype)


class TestRot90:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_k1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.rot90(a_cp),
                  np.ascontiguousarray(np.rot90(a_np)), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_k_values(self, dtype, k):
        a_np = make_arg((3, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.rot90(a_cp, k=k),
                  np.ascontiguousarray(np.rot90(a_np, k=k)), dtype=dtype)


# ====================================================================
# Roll
# ====================================================================
# Ref: cupy_tests/manipulation_tests/test_shift.py

class TestRoll:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", MANIP_SHAPES)
    def test_int_shift(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.roll(a_cp, 2), np.roll(a_np, 2), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_negative_shift(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.roll(a_cp, -2), np.roll(a_np, -2), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_with_axis(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.roll(a_cp, 1, axis=1), np.roll(a_np, 1, axis=1), dtype=dtype)


# ====================================================================
# Split (split, array_split, hsplit, vsplit, dsplit)
# ====================================================================
# Ref: cupy_tests/manipulation_tests/test_split.py

class TestSplit:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_equal_1d(self, dtype):
        a_np = make_arg((6,), dtype)
        a_cp = cp.array(a_np)
        results = cp.split(a_cp, 3)
        expected = np.split(a_np, 3)
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            assert_eq(r, e, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_indices_1d(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        results = cp.split(a_cp, [2, 4])
        expected = np.split(a_np, [2, 4])
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            assert_eq(r, e, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d_axis0(self, dtype):
        a_np = make_arg((4, 3), dtype)
        a_cp = cp.array(a_np)
        results = cp.split(a_cp, 2, axis=0)
        expected = np.split(a_np, 2, axis=0)
        for r, e in zip(results, expected):
            assert_eq(r, e, dtype=dtype)


class TestArraySplit:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_unequal(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        results = cp.array_split(a_cp, 3)
        expected = np.array_split(a_np, 3)
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            assert_eq(r, e, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_equal(self, dtype):
        a_np = make_arg((6,), dtype)
        a_cp = cp.array(a_np)
        results = cp.array_split(a_cp, 3)
        expected = np.array_split(a_np, 3)
        for r, e in zip(results, expected):
            assert_eq(r, e, dtype=dtype)


class TestHsplit:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((2, 6), dtype)
        a_cp = cp.array(a_np)
        results = cp.hsplit(a_cp, 3)
        expected = np.hsplit(a_np, 3)
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            assert_eq(r, e, dtype=dtype)


class TestVsplit:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((4, 3), dtype)
        a_cp = cp.array(a_np)
        results = cp.vsplit(a_cp, 2)
        expected = np.vsplit(a_np, 2)
        for r, e in zip(results, expected):
            assert_eq(r, e, dtype=dtype)


class TestDsplit:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        results = cp.dsplit(a_cp, 2)
        expected = np.dsplit(a_np, 2)
        for r, e in zip(results, expected):
            assert_eq(r, e, dtype=dtype)


# ====================================================================
# Join (concatenate, stack, vstack, hstack, dstack, column_stack, concat)
# ====================================================================
# Ref: cupy_tests/manipulation_tests/test_join.py

class TestConcatenate:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        a = make_arg((3,), dtype)
        b = make_arg((3,), dtype)
        assert_eq(cp.concatenate([cp.array(a), cp.array(b)]),
                  np.concatenate([a, b]), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d_axis0(self, dtype):
        a = make_arg((2, 3), dtype)
        b = make_arg((2, 3), dtype)
        assert_eq(cp.concatenate([cp.array(a), cp.array(b)], axis=0),
                  np.concatenate([a, b], axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d_axis1(self, dtype):
        a = make_arg((2, 3), dtype)
        b = make_arg((2, 3), dtype)
        assert_eq(cp.concatenate([cp.array(a), cp.array(b)], axis=1),
                  np.concatenate([a, b], axis=1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_three_arrays(self, dtype):
        a = make_arg((2,), dtype)
        b = make_arg((3,), dtype)
        c = make_arg((4,), dtype)
        assert_eq(cp.concatenate([cp.array(a), cp.array(b), cp.array(c)]),
                  np.concatenate([a, b, c]), dtype=dtype)


class TestStack:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        a = make_arg((3,), dtype)
        b = make_arg((3,), dtype)
        assert_eq(cp.stack([cp.array(a), cp.array(b)]),
                  np.stack([a, b]), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis1(self, dtype):
        a = make_arg((3,), dtype)
        b = make_arg((3,), dtype)
        assert_eq(cp.stack([cp.array(a), cp.array(b)], axis=1),
                  np.stack([a, b], axis=1), dtype=dtype)


class TestVstack:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        a = make_arg((3,), dtype)
        b = make_arg((3,), dtype)
        assert_eq(cp.vstack([cp.array(a), cp.array(b)]),
                  np.vstack([a, b]), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        a = make_arg((2, 3), dtype)
        b = make_arg((2, 3), dtype)
        assert_eq(cp.vstack([cp.array(a), cp.array(b)]),
                  np.vstack([a, b]), dtype=dtype)


class TestHstack:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        a = make_arg((3,), dtype)
        b = make_arg((3,), dtype)
        assert_eq(cp.hstack([cp.array(a), cp.array(b)]),
                  np.hstack([a, b]), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        a = make_arg((2, 3), dtype)
        b = make_arg((2, 3), dtype)
        assert_eq(cp.hstack([cp.array(a), cp.array(b)]),
                  np.hstack([a, b]), dtype=dtype)


class TestDstack:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        a = make_arg((3,), dtype)
        b = make_arg((3,), dtype)
        assert_eq(cp.dstack([cp.array(a), cp.array(b)]),
                  np.dstack([a, b]), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        a = make_arg((2, 3), dtype)
        b = make_arg((2, 3), dtype)
        assert_eq(cp.dstack([cp.array(a), cp.array(b)]),
                  np.dstack([a, b]), dtype=dtype)


class TestColumnStack:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        a = make_arg((3,), dtype)
        b = make_arg((3,), dtype)
        assert_eq(cp.column_stack([cp.array(a), cp.array(b)]),
                  np.column_stack([a, b]), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        a = make_arg((2, 3), dtype)
        b = make_arg((2, 3), dtype)
        assert_eq(cp.column_stack([cp.array(a), cp.array(b)]),
                  np.column_stack([a, b]), dtype=dtype)


class TestConcat:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        a = make_arg((3,), dtype)
        b = make_arg((3,), dtype)
        assert_eq(cp.concat([cp.array(a), cp.array(b)]),
                  np.concatenate([a, b]), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis1(self, dtype):
        a = make_arg((2, 3), dtype)
        b = make_arg((2, 3), dtype)
        assert_eq(cp.concat([cp.array(a), cp.array(b)], axis=1),
                  np.concatenate([a, b], axis=1), dtype=dtype)


# ====================================================================
# Axis (moveaxis, swapaxes, rollaxis)
# ====================================================================
# Ref: cupy_tests/manipulation_tests/test_transpose.py

class TestMoveaxis:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        result = cp.moveaxis(a_cp, 0, -1)
        expected = np.moveaxis(a_np, 0, -1)
        assert result.shape == expected.shape
        assert_eq(result, np.ascontiguousarray(expected), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_swap(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        result = cp.moveaxis(a_cp, 2, 0)
        expected = np.moveaxis(a_np, 2, 0)
        assert result.shape == expected.shape
        assert_eq(result, np.ascontiguousarray(expected), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_noop(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        result = cp.moveaxis(a_cp, 1, 1)
        assert result.shape == a_np.shape
        assert_eq(result, a_np, dtype=dtype)


class TestSwapaxes:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        result = cp.swapaxes(a_cp, 0, 1)
        expected = np.swapaxes(a_np, 0, 1)
        assert_eq(result, np.ascontiguousarray(expected), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_3d(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        result = cp.swapaxes(a_cp, 0, 2)
        expected = np.swapaxes(a_np, 0, 2)
        assert_eq(result, np.ascontiguousarray(expected), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_noop(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        result = cp.swapaxes(a_cp, 1, 1)
        assert_eq(result, a_np, dtype=dtype)


class TestRollaxis:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((3, 4, 5), dtype)
        a_cp = cp.array(a_np)
        result = cp.rollaxis(a_cp, 2, 0)
        expected = np.rollaxis(a_np, 2, 0)
        assert result.shape == expected.shape
        assert_eq(result, np.ascontiguousarray(expected), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_to_end(self, dtype):
        a_np = make_arg((3, 4, 5), dtype)
        a_cp = cp.array(a_np)
        result = cp.rollaxis(a_cp, 0, 3)
        expected = np.rollaxis(a_np, 0, 3)
        assert result.shape == expected.shape
        assert_eq(result, np.ascontiguousarray(expected), dtype=dtype)


# ====================================================================
# Broadcast (broadcast_to, broadcast_arrays)
# ====================================================================
# Ref: cupy_tests/manipulation_tests/test_broadcast.py

class TestBroadcastTo:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("src_shape,target_shape", BROADCAST_PAIRS)
    def test_basic(self, dtype, src_shape, target_shape):
        # Broadcast the smaller shape to the larger
        a_np = make_arg(target_shape, dtype)
        a_cp = cp.array(a_np)
        out_shape = np.broadcast_shapes(src_shape, target_shape)
        result = cp.broadcast_to(a_cp, out_shape)
        expected = np.broadcast_to(a_np, out_shape)
        assert result.shape == expected.shape
        assert_eq(result, np.ascontiguousarray(expected), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d_to_2d(self, dtype):
        a_np = make_arg((3,), dtype)
        a_cp = cp.array(a_np)
        result = cp.broadcast_to(a_cp, (2, 3))
        expected = np.broadcast_to(a_np, (2, 3))
        assert_eq(result, np.ascontiguousarray(expected), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_scalar_like(self, dtype):
        a_np = make_arg((1,), dtype)
        a_cp = cp.array(a_np)
        result = cp.broadcast_to(a_cp, (3, 4))
        expected = np.broadcast_to(a_np, (3, 4))
        assert_eq(result, np.ascontiguousarray(expected), dtype=dtype)


class TestBroadcastArrays:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((3, 1), dtype)
        b_np = make_arg((1, 4), dtype)
        a_cp, b_cp = cp.array(a_np), cp.array(b_np)
        ra, rb = cp.broadcast_arrays(a_cp, b_cp)
        ea, eb = np.broadcast_arrays(a_np, b_np)
        assert_eq(ra, np.ascontiguousarray(ea), dtype=dtype)
        assert_eq(rb, np.ascontiguousarray(eb), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_same_shape(self, dtype):
        a_np = make_arg((2, 3), dtype)
        b_np = make_arg((2, 3), dtype)
        a_cp, b_cp = cp.array(a_np), cp.array(b_np)
        ra, rb = cp.broadcast_arrays(a_cp, b_cp)
        assert ra.shape == (2, 3)
        assert rb.shape == (2, 3)


# ====================================================================
# Atleast (atleast_1d, atleast_2d, atleast_3d)
# ====================================================================
# Ref: cupy_tests/manipulation_tests/test_dims.py

class TestAtleast1d:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_scalar(self, dtype):
        a_np = np.array(1, dtype=dtype)
        a_cp = cp.array(a_np)
        result = cp.atleast_1d(a_cp)
        expected = np.atleast_1d(a_np)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_1d(self, dtype):
        a_np = make_arg((3,), dtype)
        a_cp = cp.array(a_np)
        result = cp.atleast_1d(a_cp)
        assert result.shape == (3,)
        assert_eq(result, a_np, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_2d(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        result = cp.atleast_1d(a_cp)
        assert result.shape == (2, 3)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_multi(self, dtype):
        a_np = np.array(1, dtype=dtype)
        b_np = make_arg((3,), dtype)
        r = cp.atleast_1d(cp.array(a_np), cp.array(b_np))
        assert isinstance(r, list)
        assert len(r) == 2


class TestAtleast2d:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_1d(self, dtype):
        a_np = make_arg((3,), dtype)
        a_cp = cp.array(a_np)
        result = cp.atleast_2d(a_cp)
        expected = np.atleast_2d(a_np)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_2d_noop(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        result = cp.atleast_2d(a_cp)
        assert result.shape == (2, 3)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_3d_noop(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        result = cp.atleast_2d(a_cp)
        assert result.shape == (2, 3, 4)


class TestAtleast3d:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_1d(self, dtype):
        a_np = make_arg((3,), dtype)
        a_cp = cp.array(a_np)
        result = cp.atleast_3d(a_cp)
        expected = np.atleast_3d(a_np)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_2d(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        result = cp.atleast_3d(a_cp)
        expected = np.atleast_3d(a_np)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_3d_noop(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        result = cp.atleast_3d(a_cp)
        assert result.shape == (2, 3, 4)


# ====================================================================
# Misc (delete, append, resize, trim_zeros, copyto, pad)
# ====================================================================
# Ref: cupy_tests/manipulation_tests/

class TestDelete:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.delete(a_cp, [1, 3]),
                  np.delete(a_np, [1, 3]), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis0(self, dtype):
        a_np = make_arg((3, 4), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.delete(a_cp, 1, axis=0),
                  np.delete(a_np, 1, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_single_index(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.delete(a_cp, 2), np.delete(a_np, 2), dtype=dtype)


class TestAppend:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        a_np = make_arg((3,), dtype)
        v_np = np.array([10, 20], dtype=dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.append(a_cp, v_np), np.append(a_np, v_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis0(self, dtype):
        a_np = make_arg((2, 3), dtype)
        v_np = make_arg((1, 3), dtype)
        a_cp = cp.array(a_np)
        v_cp = cp.array(v_np)
        assert_eq(cp.append(a_cp, v_cp, axis=0),
                  np.append(a_np, v_np, axis=0), dtype=dtype)


class TestResize:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((3,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.resize(a_cp, (2, 3)),
                  np.resize(a_np, (2, 3)), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_larger(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.resize(a_cp, (4, 3)),
                  np.resize(a_np, (4, 3)), dtype=dtype)


class TestTrimZeros:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_both(self, dtype):
        a_np = np.array([0, 0, 1, 2, 3, 0, 0], dtype=dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.trim_zeros(a_cp), np.trim_zeros(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_front(self, dtype):
        a_np = np.array([0, 0, 1, 2, 3], dtype=dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.trim_zeros(a_cp, 'f'), np.trim_zeros(a_np, 'f'), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_back(self, dtype):
        a_np = np.array([1, 2, 3, 0, 0], dtype=dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.trim_zeros(a_cp, 'b'), np.trim_zeros(a_np, 'b'), dtype=dtype)


class TestCopyto:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        dst = cp.zeros(5, dtype=dtype)
        src_np = make_arg((5,), dtype)
        src = cp.array(src_np)
        cp.copyto(dst, src)
        assert_eq(dst, src_np, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        dst = cp.zeros((2, 3), dtype=dtype)
        src_np = make_arg((2, 3), dtype)
        src = cp.array(src_np)
        cp.copyto(dst, src)
        assert_eq(dst, src_np, dtype=dtype)


class TestPad:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_constant_1d(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.pad(a_cp, 2, mode='constant', constant_values=0),
                  np.pad(a_np, 2, mode='constant', constant_values=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_constant_2d(self, dtype):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.pad(a_cp, ((1, 1), (2, 2)), mode='constant'),
                  np.pad(a_np, ((1, 1), (2, 2)), mode='constant'), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_edge_1d(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.pad(a_cp, 2, mode='edge'),
                  np.pad(a_np, 2, mode='edge'), dtype=dtype)
