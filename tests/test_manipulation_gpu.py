"""Tests verifying GPU-accelerated manipulation operations."""
import numpy as np
import numpy.testing as npt
import pytest
import macmetalpy as mp

class TestManipulationGPU:
    def test_flip_1d(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = mp.flip(mp.array(np_a))
        npt.assert_allclose(result.get(), np.flip(np_a))

    def test_flip_2d_axis0(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = mp.flip(mp.array(np_a), axis=0)
        npt.assert_allclose(result.get(), np.flip(np_a, axis=0))

    def test_fliplr(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = mp.fliplr(mp.array(np_a))
        npt.assert_allclose(result.get(), np.fliplr(np_a))

    def test_flipud(self):
        np_a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        result = mp.flipud(mp.array(np_a))
        npt.assert_allclose(result.get(), np.flipud(np_a))

    def test_roll_1d(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = mp.roll(mp.array(np_a), 2)
        npt.assert_allclose(result.get(), np.roll(np_a, 2))

    def test_roll_negative(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = mp.roll(mp.array(np_a), -2)
        npt.assert_allclose(result.get(), np.roll(np_a, -2))

    def test_tile_1d(self):
        np_a = np.array([1, 2, 3], dtype=np.float32)
        result = mp.tile(mp.array(np_a), 3)
        npt.assert_allclose(result.get(), np.tile(np_a, 3))

    def test_repeat_1d(self):
        np_a = np.array([1, 2, 3], dtype=np.float32)
        result = mp.repeat(mp.array(np_a), 3)
        npt.assert_allclose(result.get(), np.repeat(np_a, 3))

    def test_rot90_k1(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = mp.rot90(mp.array(np_a))
        npt.assert_allclose(result.get(), np.rot90(np_a))

    def test_rot90_k2(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = mp.rot90(mp.array(np_a), k=2)
        npt.assert_allclose(result.get(), np.rot90(np_a, k=2))

    def test_rot90_k3(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = mp.rot90(mp.array(np_a), k=3)
        npt.assert_allclose(result.get(), np.rot90(np_a, k=3))

    def test_broadcast_to_scalar(self):
        a = mp.array(np.array([5], dtype=np.float32))
        result = mp.broadcast_to(a, (3, 4))
        expected = np.broadcast_to(np.array([5]), (3, 4))
        npt.assert_allclose(result.get(), expected)

    def test_broadcast_to_1d(self):
        np_a = np.array([1, 2, 3], dtype=np.float32)
        result = mp.broadcast_to(mp.array(np_a), (2, 3))
        expected = np.broadcast_to(np_a, (2, 3))
        npt.assert_allclose(result.get(), expected)

    def test_pad_1d_constant(self):
        np_a = np.array([1, 2, 3], dtype=np.float32)
        result = mp.pad(mp.array(np_a), (2, 3), mode='constant', constant_values=0)
        expected = np.pad(np_a, (2, 3), mode='constant', constant_values=0)
        npt.assert_allclose(result.get(), expected)

    def test_pad_1d_nonzero_fill(self):
        np_a = np.array([1, 2, 3], dtype=np.float32)
        result = mp.pad(mp.array(np_a), 1, mode='constant', constant_values=9)
        expected = np.pad(np_a, 1, mode='constant', constant_values=9)
        npt.assert_allclose(result.get(), expected)
