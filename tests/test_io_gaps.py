"""Tests for macmetalpy.io module: save, load, savez, savez_compressed.

TDD: these tests are written FIRST, then the implementation in io.py.
"""

import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy import io as cpio


# ======================================================================
# save / load roundtrip
# ======================================================================

class TestSaveLoad:
    def test_roundtrip_1d(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
        try:
            cpio.save(path, a)
            loaded = cpio.load(path)
            npt.assert_array_equal(loaded.get(), a.get())
            assert loaded.dtype == a.dtype
        finally:
            os.unlink(path)

    def test_roundtrip_2d(self):
        a = cp.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
        try:
            cpio.save(path, a)
            loaded = cpio.load(path)
            npt.assert_array_equal(loaded.get(), a.get())
            assert loaded.shape == (2, 3)
        finally:
            os.unlink(path)

    def test_roundtrip_float16(self):
        a = cp.array([0.5, 1.5, 2.5], dtype=np.float16)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
        try:
            cpio.save(path, a)
            loaded = cpio.load(path)
            npt.assert_array_equal(loaded.get(), a.get())
            assert loaded.dtype == np.float16
        finally:
            os.unlink(path)

    def test_auto_npy_extension(self):
        """np.save auto-appends .npy if missing."""
        a = cp.array([1.0, 2.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
            path = f.name
        try:
            cpio.save(path, a)
            actual_path = path + '.npy' if not path.endswith('.npy') else path
            loaded = cpio.load(actual_path)
            npt.assert_array_equal(loaded.get(), a.get())
        finally:
            for p in [path, path + '.npy']:
                if os.path.exists(p):
                    os.unlink(p)


# ======================================================================
# savez / load roundtrip
# ======================================================================

class TestSavezLoad:
    def test_roundtrip_positional(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = cp.array([4, 5, 6], dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            cpio.savez(path, a, b)
            loaded = cpio.load(path)
            assert isinstance(loaded, dict)
            npt.assert_array_equal(loaded['arr_0'].get(), a.get())
            npt.assert_array_equal(loaded['arr_1'].get(), b.get())
        finally:
            # savez may add .npz extension
            for p in [path, path + '.npz']:
                if os.path.exists(p):
                    os.unlink(p)

    def test_roundtrip_keyword(self):
        x = cp.array([10, 20], dtype=np.int32)
        y = cp.array([0.1, 0.2], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            cpio.savez(path, x=x, y=y)
            loaded = cpio.load(path)
            assert isinstance(loaded, dict)
            npt.assert_array_equal(loaded['x'].get(), x.get())
            npt.assert_array_equal(loaded['y'].get(), y.get())
        finally:
            for p in [path, path + '.npz']:
                if os.path.exists(p):
                    os.unlink(p)


# ======================================================================
# savez_compressed / load roundtrip
# ======================================================================

class TestSavezCompressed:
    def test_roundtrip(self):
        a = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            cpio.savez_compressed(path, data=a)
            loaded = cpio.load(path)
            assert isinstance(loaded, dict)
            npt.assert_array_equal(loaded['data'].get(), a.get())
        finally:
            for p in [path, path + '.npz']:
                if os.path.exists(p):
                    os.unlink(p)

    def test_compressed_smaller(self):
        """Compressed file should be no larger than uncompressed for repetitive data."""
        a = cp.zeros(10000, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path_compressed = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path_uncompressed = f.name
        try:
            cpio.savez_compressed(path_compressed, a=a)
            cpio.savez(path_uncompressed, a=a)
            # Find the actual files (may have .npz appended)
            comp_path = path_compressed + '.npz' if os.path.exists(path_compressed + '.npz') else path_compressed
            uncomp_path = path_uncompressed + '.npz' if os.path.exists(path_uncompressed + '.npz') else path_uncompressed
            assert os.path.getsize(comp_path) <= os.path.getsize(uncomp_path)
        finally:
            for p in [path_compressed, path_compressed + '.npz',
                       path_uncompressed, path_uncompressed + '.npz']:
                if os.path.exists(p):
                    os.unlink(p)
