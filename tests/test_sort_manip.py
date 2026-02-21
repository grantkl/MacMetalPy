"""Tests for sorting and array manipulation functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp
from macmetalpy.sorting import sort, argsort, unique, searchsorted
from macmetalpy.manipulation import (
    tile, repeat, flip, roll, split, array_split,
    squeeze, ravel, moveaxis, swapaxes, broadcast_to,
)


# ------------------------------------------------------------------ Sorting


class TestSort:
    def test_sort_1d(self):
        np_arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = sort(gpu)
        expected = np.sort(np_arr)
        assert_array_equal(result.get(), expected)

    def test_sort_2d_default_axis(self):
        np_arr = np.array([[3.0, 1.0], [4.0, 2.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = sort(gpu)
        expected = np.sort(np_arr, axis=-1)
        assert_array_equal(result.get(), expected)

    def test_sort_2d_axis0(self):
        np_arr = np.array([[3.0, 1.0], [2.0, 4.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = sort(gpu, axis=0)
        expected = np.sort(np_arr, axis=0)
        assert_array_equal(result.get(), expected)

    def test_sort_already_sorted(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = sort(gpu)
        assert_array_equal(result.get(), np_arr)

    def test_sort_from_list(self):
        result = sort([5.0, 3.0, 1.0, 4.0, 2.0])
        expected = np.sort(np.array([5.0, 3.0, 1.0, 4.0, 2.0]))
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestArgsort:
    def test_argsort_1d(self):
        np_arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = argsort(gpu)
        expected = np.argsort(np_arr)
        assert_array_equal(result.get(), expected)

    def test_argsort_2d_default_axis(self):
        np_arr = np.array([[3.0, 1.0], [4.0, 2.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = argsort(gpu)
        expected = np.argsort(np_arr, axis=-1)
        assert_array_equal(result.get(), expected)

    def test_argsort_2d_axis0(self):
        np_arr = np.array([[3.0, 1.0], [2.0, 4.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = argsort(gpu, axis=0)
        expected = np.argsort(np_arr, axis=0)
        assert_array_equal(result.get(), expected)

    def test_argsort_from_list(self):
        result = argsort([5.0, 3.0, 1.0])
        expected = np.argsort(np.array([5.0, 3.0, 1.0]))
        assert_array_equal(result.get(), expected)


class TestUnique:
    def test_unique_with_duplicates(self):
        np_arr = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 2.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = unique(gpu)
        expected = np.unique(np_arr)
        assert_array_equal(result.get(), expected)

    def test_unique_already_unique(self):
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = unique(gpu)
        expected = np.unique(np_arr)
        assert_array_equal(result.get(), expected)

    def test_unique_single_element(self):
        np_arr = np.array([5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = unique(gpu)
        assert_array_equal(result.get(), np.array([5.0], dtype=np.float32))

    def test_unique_from_list(self):
        result = unique([3.0, 1.0, 1.0, 2.0])
        expected = np.unique(np.array([3.0, 1.0, 1.0, 2.0]))
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestSearchsorted:
    def test_searchsorted_left(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        values = cp.array(np.array([2.5, 0.5, 5.5], dtype=np.float32))
        result = searchsorted(gpu, values, side='left')
        expected = np.searchsorted(np_arr, np.array([2.5, 0.5, 5.5], dtype=np.float32), side='left')
        assert_array_equal(result.get(), expected)

    def test_searchsorted_right(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        values = cp.array(np.array([2.0, 3.0], dtype=np.float32))
        result = searchsorted(gpu, values, side='right')
        expected = np.searchsorted(np_arr, np.array([2.0, 3.0], dtype=np.float32), side='right')
        assert_array_equal(result.get(), expected)

    def test_searchsorted_scalar(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = searchsorted(gpu, 3.5)
        expected = np.searchsorted(np_arr, 3.5)
        assert_array_equal(result.get(), expected)

    def test_searchsorted_from_list(self):
        result = searchsorted([1.0, 2.0, 3.0], [1.5, 2.5])
        expected = np.searchsorted([1.0, 2.0, 3.0], [1.5, 2.5])
        assert_array_equal(result.get(), expected)


# ------------------------------------------------------------------ Manipulation


class TestTile:
    def test_tile_1d(self):
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = tile(gpu, 3)
        expected = np.tile(np_arr, 3)
        assert_array_equal(result.get(), expected)

    def test_tile_2d(self):
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = tile(gpu, (2, 3))
        expected = np.tile(np_arr, (2, 3))
        assert_array_equal(result.get(), expected)

    def test_tile_from_list(self):
        result = tile([1.0, 2.0], 2)
        expected = np.tile(np.array([1.0, 2.0]), 2)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestRepeat:
    def test_repeat_1d(self):
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = repeat(gpu, 2)
        expected = np.repeat(np_arr, 2)
        assert_array_equal(result.get(), expected)

    def test_repeat_with_axis(self):
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = repeat(gpu, 3, axis=1)
        expected = np.repeat(np_arr, 3, axis=1)
        assert_array_equal(result.get(), expected)

    def test_repeat_no_axis_2d(self):
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = repeat(gpu, 2)
        expected = np.repeat(np_arr, 2)
        assert_array_equal(result.get(), expected)

    def test_repeat_from_list(self):
        result = repeat([1.0, 2.0, 3.0], 2)
        expected = np.repeat(np.array([1.0, 2.0, 3.0]), 2)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestFlip:
    def test_flip_1d(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = flip(gpu)
        expected = np.flip(np_arr)
        assert_array_equal(result.get(), expected)

    def test_flip_2d_no_axis(self):
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = flip(gpu)
        expected = np.flip(np_arr)
        assert_array_equal(result.get(), expected)

    def test_flip_2d_axis0(self):
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = flip(gpu, axis=0)
        expected = np.flip(np_arr, axis=0)
        assert_array_equal(result.get(), expected)

    def test_flip_2d_axis1(self):
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = flip(gpu, axis=1)
        expected = np.flip(np_arr, axis=1)
        assert_array_equal(result.get(), expected)

    def test_flip_from_list(self):
        result = flip([1.0, 2.0, 3.0])
        expected = np.flip(np.array([1.0, 2.0, 3.0]))
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestRoll:
    def test_roll_1d_positive(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = roll(gpu, 2)
        expected = np.roll(np_arr, 2)
        assert_array_equal(result.get(), expected)

    def test_roll_1d_negative(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = roll(gpu, -2)
        expected = np.roll(np_arr, -2)
        assert_array_equal(result.get(), expected)

    def test_roll_2d_with_axis(self):
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = roll(gpu, 1, axis=1)
        expected = np.roll(np_arr, 1, axis=1)
        assert_array_equal(result.get(), expected)

    def test_roll_from_list(self):
        result = roll([1.0, 2.0, 3.0], 1)
        expected = np.roll(np.array([1.0, 2.0, 3.0]), 1)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestSplit:
    def test_split_equal(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        results = split(gpu, 3)
        expected = np.split(np_arr, 3)
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            assert_array_equal(r.get(), e)

    def test_split_indices(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        results = split(gpu, [2, 4])
        expected = np.split(np_arr, [2, 4])
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            assert_array_equal(r.get(), e)

    def test_split_2d_axis0(self):
        np_arr = np.arange(12.0, dtype=np.float32).reshape(4, 3)
        gpu = cp.array(np_arr)
        results = split(gpu, 2, axis=0)
        expected = np.split(np_arr, 2, axis=0)
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            assert_array_equal(r.get(), e)

    def test_split_from_list(self):
        results = split([1.0, 2.0, 3.0, 4.0], 2)
        expected = np.split(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        for r, e in zip(results, expected):
            assert_allclose(r.get(), e, rtol=1e-5)


class TestArraySplit:
    def test_array_split_unequal(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        results = array_split(gpu, 3)
        expected = np.array_split(np_arr, 3)
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            assert_array_equal(r.get(), e)

    def test_array_split_equal(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        results = array_split(gpu, 3)
        expected = np.array_split(np_arr, 3)
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            assert_array_equal(r.get(), e)

    def test_array_split_from_list(self):
        results = array_split([1.0, 2.0, 3.0, 4.0, 5.0], 2)
        expected = np.array_split(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2)
        for r, e in zip(results, expected):
            assert_allclose(r.get(), e, rtol=1e-5)


class TestSqueeze:
    def test_squeeze_all(self):
        np_arr = np.array([[[1.0], [2.0]]], dtype=np.float32)  # shape (1,2,1)
        gpu = cp.array(np_arr)
        result = squeeze(gpu)
        expected = np.squeeze(np_arr)
        assert result.shape == expected.shape
        assert_array_equal(result.get(), expected)

    def test_squeeze_specific_axis(self):
        np_arr = np.ones((1, 3, 1, 4), dtype=np.float32)
        gpu = cp.array(np_arr)
        result = squeeze(gpu, axis=0)
        expected = np.squeeze(np_arr, axis=0)
        assert result.shape == expected.shape

    def test_squeeze_invalid_axis(self):
        np_arr = np.ones((2, 3), dtype=np.float32)
        gpu = cp.array(np_arr)
        with pytest.raises(ValueError):
            squeeze(gpu, axis=0)

    def test_squeeze_from_list(self):
        result = squeeze([[[1.0, 2.0, 3.0]]])
        assert result.shape == (3,)


class TestRavel:
    def test_ravel_2d(self):
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = ravel(gpu)
        expected = np_arr.ravel()
        assert result.shape == expected.shape
        assert_array_equal(result.get(), expected)

    def test_ravel_3d(self):
        np_arr = np.arange(24.0, dtype=np.float32).reshape(2, 3, 4)
        gpu = cp.array(np_arr)
        result = ravel(gpu)
        expected = np_arr.ravel()
        assert result.shape == expected.shape
        assert_array_equal(result.get(), expected)

    def test_ravel_from_list(self):
        result = ravel([[1.0, 2.0], [3.0, 4.0]])
        expected = np.array([[1.0, 2.0], [3.0, 4.0]]).ravel()
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestMoveaxis:
    def test_moveaxis_3d(self):
        np_arr = np.arange(24.0, dtype=np.float32).reshape(2, 3, 4)
        gpu = cp.array(np_arr)
        result = moveaxis(gpu, 0, -1)
        expected = np.moveaxis(np_arr, 0, -1)
        assert result.shape == expected.shape
        assert_array_equal(result.get(), expected)

    def test_moveaxis_swap(self):
        np_arr = np.arange(24.0, dtype=np.float32).reshape(2, 3, 4)
        gpu = cp.array(np_arr)
        result = moveaxis(gpu, 2, 0)
        expected = np.moveaxis(np_arr, 2, 0)
        assert result.shape == expected.shape
        assert_array_equal(result.get(), expected)

    def test_moveaxis_from_list(self):
        result = moveaxis([[[1.0, 2.0], [3.0, 4.0]]], 0, -1)
        expected = np.moveaxis(np.array([[[1.0, 2.0], [3.0, 4.0]]]), 0, -1)
        assert result.shape == expected.shape
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestSwapaxes:
    def test_swapaxes_2d(self):
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = swapaxes(gpu, 0, 1)
        expected = np.swapaxes(np_arr, 0, 1)
        assert result.shape == expected.shape
        assert_array_equal(result.get(), expected)

    def test_swapaxes_3d(self):
        np_arr = np.arange(24.0, dtype=np.float32).reshape(2, 3, 4)
        gpu = cp.array(np_arr)
        result = swapaxes(gpu, 0, 2)
        expected = np.swapaxes(np_arr, 0, 2)
        assert result.shape == expected.shape
        assert_array_equal(result.get(), expected)

    def test_swapaxes_from_list(self):
        result = swapaxes([[1.0, 2.0], [3.0, 4.0]], 0, 1)
        expected = np.swapaxes(np.array([[1.0, 2.0], [3.0, 4.0]]), 0, 1)
        assert result.shape == expected.shape
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestBroadcastTo:
    def test_broadcast_to_1d(self):
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = broadcast_to(gpu, (3, 3))
        expected = np.broadcast_to(np_arr, (3, 3))
        assert result.shape == expected.shape
        assert_array_equal(result.get(), expected)

    def test_broadcast_to_scalar(self):
        np_arr = np.array([5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = broadcast_to(gpu, (2, 3))
        expected = np.broadcast_to(np_arr, (2, 3))
        assert result.shape == expected.shape
        assert_array_equal(result.get(), expected)

    def test_broadcast_to_2d(self):
        np_arr = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = broadcast_to(gpu, (3, 4))
        expected = np.broadcast_to(np_arr, (3, 4))
        assert result.shape == expected.shape
        assert_array_equal(result.get(), expected)

    def test_broadcast_to_from_list(self):
        result = broadcast_to([1.0, 2.0], (3, 2))
        expected = np.broadcast_to(np.array([1.0, 2.0]), (3, 2))
        assert result.shape == expected.shape
        assert_allclose(result.get(), expected, rtol=1e-5)
