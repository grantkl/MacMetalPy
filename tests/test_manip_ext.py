"""Tests for extended manipulation functions."""

import numpy as np
import numpy.testing as npt
import pytest

from macmetalpy.manipulation import (
    reshape, transpose, rollaxis, atleast_1d, atleast_2d, atleast_3d,
    dstack, column_stack, concat, dsplit, hsplit, vsplit, delete, append,
    resize, trim_zeros, fliplr, flipud, rot90, broadcast_arrays, copyto, pad,
)
from macmetalpy import creation


# ----- reshape / transpose / rollaxis -----

def test_reshape():
    a = creation.array(np.arange(12, dtype=np.float32))
    result = reshape(a, (3, 4))
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    npt.assert_array_equal(result.get(), expected)


def test_reshape_plain():
    result = reshape(np.arange(6, dtype=np.float32), (2, 3))
    npt.assert_array_equal(result.get(), np.arange(6, dtype=np.float32).reshape(2, 3))


def test_transpose():
    a = creation.array(np.arange(6, dtype=np.float32).reshape(2, 3))
    result = transpose(a)
    expected = np.arange(6, dtype=np.float32).reshape(2, 3).T
    npt.assert_array_equal(result.get(), expected)


def test_transpose_axes():
    a = creation.array(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    result = transpose(a, axes=(1, 2, 0))
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(1, 2, 0)
    npt.assert_array_equal(result.get(), expected)


def test_rollaxis():
    a = creation.array(np.ones((3, 4, 5), dtype=np.float32))
    result = rollaxis(a, 2, 0)
    expected = np.rollaxis(np.ones((3, 4, 5), dtype=np.float32), 2, 0)
    assert result.shape == expected.shape
    npt.assert_array_equal(result.get(), np.ascontiguousarray(expected))


# ----- atleast_*d -----

def test_atleast_1d_scalar():
    result = atleast_1d(creation.array(np.float32(5.0)))
    assert result.get().ndim >= 1
    npt.assert_array_equal(result.get(), np.atleast_1d(np.float32(5.0)))


def test_atleast_1d_multi():
    a = creation.array(np.float32(1.0))
    b = creation.array(np.array([2, 3], dtype=np.float32))
    r = atleast_1d(a, b)
    assert isinstance(r, list)
    assert len(r) == 2


def test_atleast_2d():
    a = creation.array(np.array([1, 2, 3], dtype=np.float32))
    result = atleast_2d(a)
    expected = np.atleast_2d(np.array([1, 2, 3], dtype=np.float32))
    npt.assert_array_equal(result.get(), expected)


def test_atleast_3d():
    a = creation.array(np.array([[1, 2], [3, 4]], dtype=np.float32))
    result = atleast_3d(a)
    expected = np.atleast_3d(np.array([[1, 2], [3, 4]], dtype=np.float32))
    npt.assert_array_equal(result.get(), expected)


# ----- stacking -----

def test_dstack():
    a = creation.array(np.array([1, 2, 3], dtype=np.float32))
    b = creation.array(np.array([4, 5, 6], dtype=np.float32))
    result = dstack((a, b))
    expected = np.dstack(([1, 2, 3], [4, 5, 6])).astype(np.float32)
    npt.assert_array_equal(result.get(), expected)


def test_column_stack():
    a = creation.array(np.array([1, 2, 3], dtype=np.float32))
    b = creation.array(np.array([4, 5, 6], dtype=np.float32))
    result = column_stack((a, b))
    expected = np.column_stack(([1, 2, 3], [4, 5, 6])).astype(np.float32)
    npt.assert_array_equal(result.get(), expected)


def test_concat():
    a = creation.array(np.array([1, 2], dtype=np.float32))
    b = creation.array(np.array([3, 4], dtype=np.float32))
    result = concat((a, b))
    expected = np.concatenate(([1, 2], [3, 4])).astype(np.float32)
    npt.assert_array_equal(result.get(), expected)


def test_concat_axis1():
    a = creation.array(np.ones((2, 3), dtype=np.float32))
    b = creation.array(np.zeros((2, 3), dtype=np.float32))
    result = concat((a, b), axis=1)
    expected = np.concatenate((np.ones((2, 3)), np.zeros((2, 3))), axis=1).astype(np.float32)
    npt.assert_array_equal(result.get(), expected)


# ----- splitting -----

def test_dsplit():
    a = creation.array(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    result = dsplit(a, 2)
    expected = np.dsplit(np.arange(24, dtype=np.float32).reshape(2, 3, 4), 2)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        npt.assert_array_equal(r.get(), e)


def test_hsplit():
    a = creation.array(np.arange(6, dtype=np.float32).reshape(2, 3))
    result = hsplit(a, 3)
    expected = np.hsplit(np.arange(6, dtype=np.float32).reshape(2, 3), 3)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        npt.assert_array_equal(r.get(), e)


def test_vsplit():
    a = creation.array(np.arange(12, dtype=np.float32).reshape(4, 3))
    result = vsplit(a, 2)
    expected = np.vsplit(np.arange(12, dtype=np.float32).reshape(4, 3), 2)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        npt.assert_array_equal(r.get(), e)


# ----- delete / append / resize -----

def test_delete():
    a = creation.array(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    result = delete(a, [1, 3])
    expected = np.delete(np.array([1, 2, 3, 4, 5], dtype=np.float32), [1, 3])
    npt.assert_array_equal(result.get(), expected)


def test_delete_axis():
    a = creation.array(np.arange(12, dtype=np.float32).reshape(3, 4))
    result = delete(a, 1, axis=0)
    expected = np.delete(np.arange(12, dtype=np.float32).reshape(3, 4), 1, axis=0)
    npt.assert_array_equal(result.get(), expected)


def test_append():
    a = creation.array(np.array([1, 2, 3], dtype=np.float32))
    result = append(a, [4, 5])
    expected = np.append(np.array([1, 2, 3], dtype=np.float32), [4, 5])
    npt.assert_array_equal(result.get(), expected)


def test_append_axis():
    a = creation.array(np.ones((2, 3), dtype=np.float32))
    b = creation.array(np.zeros((1, 3), dtype=np.float32))
    result = append(a, b, axis=0)
    expected = np.append(np.ones((2, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32), axis=0)
    npt.assert_array_equal(result.get(), expected)


def test_resize():
    a = creation.array(np.array([1, 2, 3], dtype=np.float32))
    result = resize(a, (2, 3))
    expected = np.resize(np.array([1, 2, 3], dtype=np.float32), (2, 3))
    npt.assert_array_equal(result.get(), expected)


# ----- trim_zeros -----

def test_trim_zeros():
    a = creation.array(np.array([0, 0, 1, 2, 0, 0], dtype=np.float32))
    result = trim_zeros(a)
    expected = np.trim_zeros(np.array([0, 0, 1, 2, 0, 0], dtype=np.float32))
    npt.assert_array_equal(result.get(), expected)


def test_trim_zeros_front():
    a = creation.array(np.array([0, 0, 1, 2, 3], dtype=np.float32))
    result = trim_zeros(a, 'f')
    expected = np.trim_zeros(np.array([0, 0, 1, 2, 3], dtype=np.float32), 'f')
    npt.assert_array_equal(result.get(), expected)


# ----- fliplr / flipud / rot90 -----

def test_fliplr():
    a = creation.array(np.arange(6, dtype=np.float32).reshape(2, 3))
    result = fliplr(a)
    expected = np.ascontiguousarray(np.fliplr(np.arange(6, dtype=np.float32).reshape(2, 3)))
    npt.assert_array_equal(result.get(), expected)


def test_flipud():
    a = creation.array(np.arange(6, dtype=np.float32).reshape(2, 3))
    result = flipud(a)
    expected = np.ascontiguousarray(np.flipud(np.arange(6, dtype=np.float32).reshape(2, 3)))
    npt.assert_array_equal(result.get(), expected)


def test_rot90():
    a = creation.array(np.arange(4, dtype=np.float32).reshape(2, 2))
    result = rot90(a)
    expected = np.ascontiguousarray(np.rot90(np.arange(4, dtype=np.float32).reshape(2, 2)))
    npt.assert_array_equal(result.get(), expected)


def test_rot90_k2():
    a = creation.array(np.arange(4, dtype=np.float32).reshape(2, 2))
    result = rot90(a, k=2)
    expected = np.ascontiguousarray(np.rot90(np.arange(4, dtype=np.float32).reshape(2, 2), k=2))
    npt.assert_array_equal(result.get(), expected)


# ----- broadcast_arrays -----

def test_broadcast_arrays():
    a = creation.array(np.array([[1], [2], [3]], dtype=np.float32))
    b = creation.array(np.array([4, 5, 6], dtype=np.float32))
    ra, rb = broadcast_arrays(a, b)
    ea, eb = np.broadcast_arrays(np.array([[1], [2], [3]], dtype=np.float32),
                                  np.array([4, 5, 6], dtype=np.float32))
    npt.assert_array_equal(ra.get(), np.ascontiguousarray(ea))
    npt.assert_array_equal(rb.get(), np.ascontiguousarray(eb))


# ----- copyto -----

def test_copyto():
    dst = creation.array(np.zeros(3, dtype=np.float32))
    src = creation.array(np.array([1, 2, 3], dtype=np.float32))
    copyto(dst, src)
    npt.assert_array_equal(dst.get(), np.array([1, 2, 3], dtype=np.float32))


# ----- pad -----

def test_pad_constant():
    a = creation.array(np.array([1, 2, 3], dtype=np.float32))
    result = pad(a, 2, mode='constant', constant_values=0)
    expected = np.pad(np.array([1, 2, 3], dtype=np.float32), 2, mode='constant', constant_values=0)
    npt.assert_array_equal(result.get(), expected)


def test_pad_2d():
    a = creation.array(np.ones((2, 3), dtype=np.float32))
    result = pad(a, ((1, 1), (2, 2)), mode='constant')
    expected = np.pad(np.ones((2, 3), dtype=np.float32), ((1, 1), (2, 2)), mode='constant')
    npt.assert_array_equal(result.get(), expected)
