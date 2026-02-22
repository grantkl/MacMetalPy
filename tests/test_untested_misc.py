"""Tests for stats/io/misc top-level APIs in macmetalpy (batch 4).

Covers: histogram, histogram2d, histogram_bin_edges, histogramdd, bincount,
digitize, convolve, correlate, corrcoef, cov, interp, piecewise, packbits,
unpackbits, savez, savez_compressed, isnat, copyto, format_float_positional,
format_float_scientific, vectorize, einsum_path, unwrap, spacing, nextafter,
ix_, broadcast_shapes, real_if_close, around.
"""

import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp


# ======================================================================
# histogram
# ======================================================================

class TestHistogram:
    def test_basic(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        h_cp, edges_cp = cp.histogram(a_cp, bins=3)
        h_np, edges_np = np.histogram(a_np, bins=3)
        npt.assert_array_equal(h_cp.get(), h_np)
        npt.assert_allclose(edges_cp.get(), edges_np.astype(np.float32), rtol=1e-5)

    def test_with_range(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        h_cp, edges_cp = cp.histogram(a_cp, bins=2, range=(2.0, 5.0))
        h_np, edges_np = np.histogram(a_np, bins=2, range=(2.0, 5.0))
        npt.assert_array_equal(h_cp.get(), h_np)
        npt.assert_allclose(edges_cp.get(), edges_np.astype(np.float32), rtol=1e-5)

    def test_uniform_data(self):
        a_np = np.arange(10, dtype=np.float32)
        a_cp = cp.array(a_np)
        h_cp, edges_cp = cp.histogram(a_cp, bins=5)
        h_np, edges_np = np.histogram(a_np, bins=5)
        npt.assert_array_equal(h_cp.get(), h_np)


# ======================================================================
# histogram2d
# ======================================================================

class TestHistogram2d:
    def test_basic(self):
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y_np = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        x_cp, y_cp = cp.array(x_np), cp.array(y_np)
        h_cp, xedges_cp, yedges_cp = cp.histogram2d(x_cp, y_cp, bins=2)
        h_np, xedges_np, yedges_np = np.histogram2d(x_np, y_np, bins=2)
        npt.assert_array_equal(h_cp.get(), h_np.astype(np.float32))

    def test_different_bins(self):
        x_np = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y_np = np.array([0.0, 1.0, 1.0, 2.0, 3.0], dtype=np.float32)
        x_cp, y_cp = cp.array(x_np), cp.array(y_np)
        h_cp, xedges_cp, yedges_cp = cp.histogram2d(x_cp, y_cp, bins=[2, 3])
        h_np, xedges_np, yedges_np = np.histogram2d(x_np, y_np, bins=[2, 3])
        npt.assert_array_equal(h_cp.get(), h_np.astype(np.float32))


# ======================================================================
# histogram_bin_edges
# ======================================================================

class TestHistogramBinEdges:
    def test_basic(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        edges_cp = cp.histogram_bin_edges(a_cp, bins=3)
        edges_np = np.histogram_bin_edges(a_np, bins=3)
        npt.assert_allclose(edges_cp.get(), edges_np.astype(np.float32), rtol=1e-5)

    def test_with_range(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        edges_cp = cp.histogram_bin_edges(a_cp, bins=4, range=(0.0, 6.0))
        edges_np = np.histogram_bin_edges(a_np, bins=4, range=(0.0, 6.0))
        npt.assert_allclose(edges_cp.get(), edges_np.astype(np.float32), rtol=1e-5)


# ======================================================================
# histogramdd
# ======================================================================

class TestHistogramdd:
    def test_basic_2d(self):
        sample_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        sample_cp = cp.array(sample_np)
        h_cp, edges_cp = cp.histogramdd(sample_cp, bins=2)
        h_np, edges_np = np.histogramdd(sample_np, bins=2)
        npt.assert_array_equal(h_cp.get(), h_np.astype(np.float32))

    def test_3_samples(self):
        rng = np.random.RandomState(42)
        sample_np = rng.rand(20, 3).astype(np.float32)
        sample_cp = cp.array(sample_np)
        h_cp, edges_cp = cp.histogramdd(sample_cp, bins=3)
        h_np, edges_np = np.histogramdd(sample_np, bins=3)
        npt.assert_array_equal(h_cp.get(), h_np.astype(np.float32))


# ======================================================================
# bincount
# ======================================================================

class TestBincount:
    def test_basic(self):
        a_np = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)
        a_cp = cp.array(a_np)
        bc_cp = cp.bincount(a_cp)
        bc_np = np.bincount(a_np)
        npt.assert_array_equal(bc_cp.get(), bc_np)

    def test_minlength(self):
        a_np = np.array([0, 1, 1], dtype=np.int32)
        a_cp = cp.array(a_np)
        bc_cp = cp.bincount(a_cp, minlength=5)
        bc_np = np.bincount(a_np, minlength=5)
        npt.assert_array_equal(bc_cp.get(), bc_np)

    def test_larger_values(self):
        a_np = np.array([3, 3, 5, 5, 5, 7], dtype=np.int32)
        a_cp = cp.array(a_np)
        bc_cp = cp.bincount(a_cp)
        bc_np = np.bincount(a_np)
        npt.assert_array_equal(bc_cp.get(), bc_np)


# ======================================================================
# digitize
# ======================================================================

class TestDigitize:
    def test_basic(self):
        x_np = np.array([0.2, 6.4, 3.0, 1.6], dtype=np.float32)
        bins_np = np.array([0.0, 1.0, 2.5, 4.0, 10.0], dtype=np.float32)
        x_cp, bins_cp = cp.array(x_np), cp.array(bins_np)
        d_cp = cp.digitize(x_cp, bins_cp)
        d_np = np.digitize(x_np, bins_np)
        npt.assert_array_equal(d_cp.get(), d_np)

    def test_right(self):
        x_np = np.array([1.0, 2.5, 4.0], dtype=np.float32)
        bins_np = np.array([1.0, 2.5, 4.0], dtype=np.float32)
        x_cp, bins_cp = cp.array(x_np), cp.array(bins_np)
        d_cp = cp.digitize(x_cp, bins_cp, right=True)
        d_np = np.digitize(x_np, bins_np, right=True)
        npt.assert_array_equal(d_cp.get(), d_np)


# ======================================================================
# convolve
# ======================================================================

class TestConvolve:
    def test_full_mode(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v_np = np.array([0.0, 1.0, 0.5], dtype=np.float32)
        a_cp, v_cp = cp.array(a_np), cp.array(v_np)
        r_cp = cp.convolve(a_cp, v_cp)
        r_np = np.convolve(a_np, v_np)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_same_mode(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        v_np = np.array([1.0, 2.0], dtype=np.float32)
        a_cp, v_cp = cp.array(a_np), cp.array(v_np)
        r_cp = cp.convolve(a_cp, v_cp, mode='same')
        r_np = np.convolve(a_np, v_np, mode='same')
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_valid_mode(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        v_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        a_cp, v_cp = cp.array(a_np), cp.array(v_np)
        r_cp = cp.convolve(a_cp, v_cp, mode='valid')
        r_np = np.convolve(a_np, v_np, mode='valid')
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)


# ======================================================================
# correlate
# ======================================================================

class TestCorrelate:
    def test_default_mode(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v_np = np.array([0.0, 1.0, 0.5], dtype=np.float32)
        a_cp, v_cp = cp.array(a_np), cp.array(v_np)
        r_cp = cp.correlate(a_cp, v_cp)
        r_np = np.correlate(a_np, v_np)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_full_mode(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v_np = np.array([1.0, 0.5], dtype=np.float32)
        a_cp, v_cp = cp.array(a_np), cp.array(v_np)
        r_cp = cp.correlate(a_cp, v_cp, mode='full')
        r_np = np.correlate(a_np, v_np, mode='full')
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)


# ======================================================================
# corrcoef
# ======================================================================

class TestCorrcoef:
    def test_basic(self):
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_np = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        x_cp, y_cp = cp.array(x_np), cp.array(y_np)
        cc_cp = cp.corrcoef(x_cp, y_cp)
        cc_np = np.corrcoef(x_np, y_np).astype(np.float32)
        npt.assert_allclose(cc_cp.get(), cc_np, rtol=1e-5)

    def test_single_array(self):
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x_cp = cp.array(x_np)
        cc_cp = cp.corrcoef(x_cp)
        cc_np = np.corrcoef(x_np).astype(np.float32)
        npt.assert_allclose(cc_cp.get(), cc_np, rtol=1e-5)

    def test_negative_correlation(self):
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_np = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        x_cp, y_cp = cp.array(x_np), cp.array(y_np)
        cc_cp = cp.corrcoef(x_cp, y_cp)
        cc_np = np.corrcoef(x_np, y_np).astype(np.float32)
        npt.assert_allclose(cc_cp.get(), cc_np, rtol=1e-5)


# ======================================================================
# cov
# ======================================================================

class TestCov:
    def test_basic(self):
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x_cp = cp.array(x_np)
        cv_cp = cp.cov(x_cp)
        cv_np = np.cov(x_np).astype(np.float32)
        npt.assert_allclose(cv_cp.get(), cv_np, rtol=1e-5)

    def test_1d(self):
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        x_cp = cp.array(x_np)
        cv_cp = cp.cov(x_cp)
        cv_np = np.cov(x_np).astype(np.float32)
        npt.assert_allclose(cv_cp.get(), cv_np, rtol=1e-5)

    def test_with_y(self):
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_np = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        x_cp, y_cp = cp.array(x_np), cp.array(y_np)
        cv_cp = cp.cov(x_cp, y_cp)
        cv_np = np.cov(x_np, y_np).astype(np.float32)
        npt.assert_allclose(cv_cp.get(), cv_np, rtol=1e-5)


# ======================================================================
# interp
# ======================================================================

class TestInterp:
    def test_basic(self):
        xp_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fp_np = np.array([3.0, 2.0, 0.0], dtype=np.float32)
        x_np = np.array([1.5, 2.5], dtype=np.float32)
        xp_cp, fp_cp, x_cp = cp.array(xp_np), cp.array(fp_np), cp.array(x_np)
        r_cp = cp.interp(x_cp, xp_cp, fp_cp)
        r_np = np.interp(x_np, xp_np, fp_np).astype(np.float32)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_extrapolation(self):
        xp_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fp_np = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        x_np = np.array([0.0, 4.0], dtype=np.float32)
        xp_cp, fp_cp, x_cp = cp.array(xp_np), cp.array(fp_np), cp.array(x_np)
        r_cp = cp.interp(x_cp, xp_cp, fp_cp)
        r_np = np.interp(x_np, xp_np, fp_np).astype(np.float32)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_single_point(self):
        xp_np = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        fp_np = np.array([0.0, 10.0, 20.0], dtype=np.float32)
        x_np = np.array([0.5], dtype=np.float32)
        xp_cp, fp_cp, x_cp = cp.array(xp_np), cp.array(fp_np), cp.array(x_np)
        r_cp = cp.interp(x_cp, xp_cp, fp_cp)
        r_np = np.interp(x_np, xp_np, fp_np).astype(np.float32)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)


# ======================================================================
# piecewise
# ======================================================================

class TestPiecewise:
    def test_basic(self):
        x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        x_cp = cp.array(x_np)
        r_cp = cp.piecewise(x_cp, [x_cp < 0, x_cp >= 0], [-1, 1])
        r_np = np.piecewise(x_np, [x_np < 0, x_np >= 0], [-1, 1])
        npt.assert_array_equal(r_cp.get(), r_np.astype(np.float32))

    def test_three_conditions(self):
        x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        x_cp = cp.array(x_np)
        r_cp = cp.piecewise(x_cp, [x_cp < 0, x_cp == 0, x_cp > 0], [-10, 0, 10])
        r_np = np.piecewise(x_np, [x_np < 0, x_np == 0, x_np > 0], [-10, 0, 10])
        npt.assert_array_equal(r_cp.get(), r_np.astype(np.float32))


# ======================================================================
# packbits / unpackbits
# ======================================================================

class TestPackbits:
    def test_basic(self):
        a_np = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.int32)
        a_cp = cp.array(a_np)
        p_cp = cp.packbits(a_cp)
        p_np = np.packbits(a_np.astype(np.uint8))
        npt.assert_array_equal(p_cp.get(), p_np.astype(np.uint16))

    def test_short_input(self):
        a_np = np.array([1, 0, 1], dtype=np.int32)
        a_cp = cp.array(a_np)
        p_cp = cp.packbits(a_cp)
        p_np = np.packbits(a_np.astype(np.uint8))
        npt.assert_array_equal(p_cp.get(), p_np.astype(np.uint16))


class TestUnpackbits:
    def test_roundtrip(self):
        a_np = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.int32)
        a_cp = cp.array(a_np)
        packed = cp.packbits(a_cp)
        unpacked = cp.unpackbits(packed)
        npt.assert_array_equal(unpacked.get()[:8], a_np)

    def test_unpackbits_basic(self):
        # Pack a known value and verify unpack matches numpy
        a_np = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.int32)
        a_cp = cp.array(a_np)
        packed = cp.packbits(a_cp)
        unpacked = cp.unpackbits(packed)
        np_packed = np.packbits(a_np.astype(np.uint8))
        np_unpacked = np.unpackbits(np_packed)
        npt.assert_array_equal(unpacked.get(), np_unpacked.astype(np.uint16))


# ======================================================================
# savez
# ======================================================================

class TestSavez:
    def test_positional_args(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = cp.array([4.0, 5.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            cp.savez(path, a, b)
            data = np.load(path)
            npt.assert_array_equal(data['arr_0'], a.get())
            npt.assert_array_equal(data['arr_1'], b.get())
        finally:
            os.unlink(path)

    def test_keyword_args(self):
        x = cp.array([10.0, 20.0], dtype=np.float32)
        y = cp.array([30.0, 40.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            cp.savez(path, x=x, y=y)
            data = np.load(path)
            npt.assert_array_equal(data['x'], x.get())
            npt.assert_array_equal(data['y'], y.get())
        finally:
            os.unlink(path)


# ======================================================================
# savez_compressed
# ======================================================================

class TestSavezCompressed:
    def test_basic(self):
        a = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            cp.savez_compressed(path, data=a)
            loaded = np.load(path)
            npt.assert_array_equal(loaded['data'], a.get())
        finally:
            os.unlink(path)

    def test_multiple_arrays(self):
        a = cp.array([1.0, 2.0], dtype=np.float32)
        b = cp.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            cp.savez_compressed(path, a, b=b)
            loaded = np.load(path)
            npt.assert_array_equal(loaded['arr_0'], a.get())
            npt.assert_array_equal(loaded['b'], b.get())
        finally:
            os.unlink(path)

    def test_compressed_smaller_than_uncompressed(self):
        # Large array with repetitive data should compress well
        a = cp.array(np.zeros(10000, dtype=np.float32))
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path_compressed = f.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path_uncompressed = f.name
        try:
            cp.savez_compressed(path_compressed, data=a)
            cp.savez(path_uncompressed, data=a)
            size_compressed = os.path.getsize(path_compressed)
            size_uncompressed = os.path.getsize(path_uncompressed)
            assert size_compressed <= size_uncompressed
        finally:
            os.unlink(path_compressed)
            os.unlink(path_uncompressed)


# ======================================================================
# isnat
# ======================================================================

class TestIsnat:
    def test_nat(self):
        result = cp.isnat(np.datetime64('NaT'))
        assert result is True or result == True

    def test_not_nat(self):
        result = cp.isnat(np.datetime64('2021-01-01'))
        assert result is False or result == False

    def test_array(self):
        a = np.array(['NaT', '2021-01-01', 'NaT'], dtype='datetime64')
        result = cp.isnat(a)
        expected = np.isnat(a)
        npt.assert_array_equal(np.asarray(result), expected)


# ======================================================================
# copyto
# ======================================================================

class TestCopyto:
    def test_basic(self):
        src = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        dst = cp.zeros(3, dtype=np.float32)
        cp.copyto(dst, src)
        npt.assert_array_equal(dst.get(), src.get())

    def test_2d(self):
        src = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        dst = cp.zeros((2, 2), dtype=np.float32)
        cp.copyto(dst, src)
        npt.assert_array_equal(dst.get(), src.get())

    def test_overwrite(self):
        dst = cp.array([10.0, 20.0, 30.0], dtype=np.float32)
        src = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        cp.copyto(dst, src)
        npt.assert_array_equal(dst.get(), src.get())


# ======================================================================
# format_float_positional
# ======================================================================

class TestFormatFloatPositional:
    def test_basic(self):
        result_cp = cp.format_float_positional(1.5)
        result_np = np.format_float_positional(1.5)
        assert result_cp == result_np

    def test_precision(self):
        result_cp = cp.format_float_positional(3.14159, precision=3)
        result_np = np.format_float_positional(3.14159, precision=3)
        assert result_cp == result_np

    def test_zero(self):
        result_cp = cp.format_float_positional(0.0)
        result_np = np.format_float_positional(0.0)
        assert result_cp == result_np


# ======================================================================
# format_float_scientific
# ======================================================================

class TestFormatFloatScientific:
    def test_basic(self):
        result_cp = cp.format_float_scientific(1.5)
        result_np = np.format_float_scientific(1.5)
        assert result_cp == result_np

    def test_precision(self):
        result_cp = cp.format_float_scientific(3.14159, precision=3)
        result_np = np.format_float_scientific(3.14159, precision=3)
        assert result_cp == result_np

    def test_large_number(self):
        result_cp = cp.format_float_scientific(1.23e10)
        result_np = np.format_float_scientific(1.23e10)
        assert result_cp == result_np


# ======================================================================
# vectorize
# ======================================================================

class TestVectorize:
    def test_basic(self):
        def my_func(x):
            return x * 2 + 1
        vf_cp = cp.vectorize(my_func)
        vf_np = np.vectorize(my_func)
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = vf_cp(a_cp)
        r_np = vf_np(a_np).astype(np.float32)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_conditional(self):
        def clamp(x):
            return max(0.0, min(x, 1.0))
        vf_cp = cp.vectorize(clamp)
        vf_np = np.vectorize(clamp)
        a_np = np.array([-0.5, 0.3, 0.7, 1.5], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = vf_cp(a_cp)
        r_np = vf_np(a_np).astype(np.float32)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)


# ======================================================================
# einsum_path
# ======================================================================

class TestEinsumPath:
    def test_basic(self):
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        a_cp, b_cp = cp.array(a_np), cp.array(b_np)
        path_cp, info_cp = cp.einsum_path('ij,jk->ik', a_cp, b_cp)
        path_np, info_np = np.einsum_path('ij,jk->ik', a_np, b_np)
        assert path_cp == path_np

    def test_three_operands(self):
        a_np = np.ones((2, 3), dtype=np.float32)
        b_np = np.ones((3, 4), dtype=np.float32)
        c_np = np.ones((4, 5), dtype=np.float32)
        a_cp, b_cp, c_cp = cp.array(a_np), cp.array(b_np), cp.array(c_np)
        path_cp, info_cp = cp.einsum_path('ij,jk,kl->il', a_cp, b_cp, c_cp)
        path_np, info_np = np.einsum_path('ij,jk,kl->il', a_np, b_np, c_np)
        assert path_cp == path_np


# ======================================================================
# unwrap
# ======================================================================

class TestUnwrap:
    def test_basic(self):
        a_np = np.array([0, 0.78, 5.49, 6.28], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = cp.unwrap(a_cp)
        r_np = np.unwrap(a_np).astype(np.float32)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_no_wrapping_needed(self):
        a_np = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = cp.unwrap(a_cp)
        r_np = np.unwrap(a_np).astype(np.float32)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_multiple_wraps(self):
        a_np = np.array([0, 3.14, 6.28, 9.42, 12.56], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = cp.unwrap(a_cp)
        r_np = np.unwrap(a_np).astype(np.float32)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)


# ======================================================================
# spacing
# ======================================================================

class TestSpacing:
    def test_one(self):
        a_np = np.array([1.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = cp.spacing(a_cp)
        r_np = np.spacing(a_np)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_multiple_values(self):
        a_np = np.array([1.0, 100.0, 1e-10], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = cp.spacing(a_cp)
        r_np = np.spacing(a_np)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)


# ======================================================================
# nextafter
# ======================================================================

class TestNextafter:
    def test_basic(self):
        x1_np = np.array([1.0], dtype=np.float32)
        x2_np = np.array([2.0], dtype=np.float32)
        x1_cp, x2_cp = cp.array(x1_np), cp.array(x2_np)
        r_cp = cp.nextafter(x1_cp, x2_cp)
        r_np = np.nextafter(x1_np, x2_np)
        npt.assert_array_equal(r_cp.get(), r_np)

    def test_toward_negative(self):
        x1_np = np.array([1.0], dtype=np.float32)
        x2_np = np.array([0.0], dtype=np.float32)
        x1_cp, x2_cp = cp.array(x1_np), cp.array(x2_np)
        r_cp = cp.nextafter(x1_cp, x2_cp)
        r_np = np.nextafter(x1_np, x2_np)
        npt.assert_array_equal(r_cp.get(), r_np)

    def test_multiple(self):
        x1_np = np.array([0.0, 1.0, 10.0], dtype=np.float32)
        x2_np = np.array([1.0, 2.0, 11.0], dtype=np.float32)
        x1_cp, x2_cp = cp.array(x1_np), cp.array(x2_np)
        r_cp = cp.nextafter(x1_cp, x2_cp)
        r_np = np.nextafter(x1_np, x2_np)
        npt.assert_array_equal(r_cp.get(), r_np)


# ======================================================================
# ix_
# ======================================================================

class TestIx:
    def test_basic(self):
        r_cp = cp.ix_(cp.array([0, 1], dtype=np.int32), cp.array([2, 3], dtype=np.int32))
        r_np = np.ix_(np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32))
        assert len(r_cp) == len(r_np)
        for c, n in zip(r_cp, r_np):
            npt.assert_array_equal(c.get(), n)

    def test_3d(self):
        r_cp = cp.ix_(
            cp.array([0, 1], dtype=np.int32),
            cp.array([2, 3, 4], dtype=np.int32),
            cp.array([5], dtype=np.int32),
        )
        r_np = np.ix_(
            np.array([0, 1], dtype=np.int32),
            np.array([2, 3, 4], dtype=np.int32),
            np.array([5], dtype=np.int32),
        )
        assert len(r_cp) == len(r_np)
        for c, n in zip(r_cp, r_np):
            npt.assert_array_equal(c.get(), n)


# ======================================================================
# broadcast_shapes
# ======================================================================

class TestBroadcastShapes:
    def test_basic(self):
        result = cp.broadcast_shapes((3, 1), (1, 4))
        expected = np.broadcast_shapes((3, 1), (1, 4))
        assert result == expected

    def test_scalar(self):
        result = cp.broadcast_shapes((3, 4), ())
        expected = np.broadcast_shapes((3, 4), ())
        assert result == expected

    def test_three_shapes(self):
        result = cp.broadcast_shapes((1, 2), (3, 1), (3, 2))
        expected = np.broadcast_shapes((1, 2), (3, 1), (3, 2))
        assert result == expected


# ======================================================================
# real_if_close
# ======================================================================

class TestRealIfClose:
    def test_real_valued_complex(self):
        a_np = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=np.complex64)
        a_cp = cp.array(a_np)
        r_cp = cp.real_if_close(a_cp)
        r_np = np.real_if_close(a_np)
        npt.assert_allclose(r_cp.get(), r_np.astype(np.float32), rtol=1e-5)

    def test_already_real(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = cp.real_if_close(a_cp)
        npt.assert_allclose(r_cp.get(), a_np, rtol=1e-5)


# ======================================================================
# around
# ======================================================================

class TestAround:
    def test_basic(self):
        a_np = np.array([1.567, 2.345, 3.789], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = cp.around(a_cp, decimals=1)
        r_np = np.around(a_np, decimals=1)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_no_decimals(self):
        a_np = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = cp.around(a_cp)
        r_np = np.around(a_np)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)

    def test_negative_decimals(self):
        a_np = np.array([123.0, 456.0, 789.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        r_cp = cp.around(a_cp, decimals=-1)
        r_np = np.around(a_np, decimals=-1)
        npt.assert_allclose(r_cp.get(), r_np, rtol=1e-5)
