"""Tests for missing parameter support in fft.py, sorting.py, and indexing.py.

Covers:
- fft.py: norm= parameter on all 14 transform functions
- sorting.py: kind=, order=, sorter= parameters
- indexing.py: out=, mode= parameters on take/choose/compress/put
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy import fft as cpfft

FFT_TOL = dict(rtol=1e-3, atol=1e-4)


# ======================================================================
# FFT norm= parameter tests
# ======================================================================

class TestFftNorm:
    """Test norm= parameter on all 14 FFT transform functions."""

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_fft_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.fft(cp.array(a), norm=norm)
        expected = np.fft.fft(a, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_ifft_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.ifft(cp.array(a), norm=norm)
        expected = np.fft.ifft(a, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_fft2_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        result = cpfft.fft2(cp.array(a), norm=norm)
        expected = np.fft.fft2(a, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_ifft2_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        result = cpfft.ifft2(cp.array(a), norm=norm)
        expected = np.fft.ifft2(a, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_fftn_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        result = cpfft.fftn(cp.array(a), norm=norm)
        expected = np.fft.fftn(a, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_ifftn_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        result = cpfft.ifftn(cp.array(a), norm=norm)
        expected = np.fft.ifftn(a, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_rfft_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.rfft(cp.array(a), norm=norm)
        expected = np.fft.rfft(a, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_irfft_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        freq = np.fft.rfft(a)
        result = cpfft.irfft(cp.array(freq), norm=norm)
        expected = np.fft.irfft(freq, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_rfft2_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        result = cpfft.rfft2(cp.array(a), norm=norm)
        expected = np.fft.rfft2(a, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_irfft2_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(8, 8).astype(np.float32)
        freq = np.fft.rfft2(a)
        result = cpfft.irfft2(cp.array(freq), norm=norm)
        expected = np.fft.irfft2(freq, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_rfftn_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        result = cpfft.rfftn(cp.array(a), norm=norm)
        expected = np.fft.rfftn(a, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_irfftn_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(4, 4, 4).astype(np.float32)
        freq = np.fft.rfftn(a)
        result = cpfft.irfftn(cp.array(freq), norm=norm)
        expected = np.fft.irfftn(freq, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_hfft_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        freq = np.fft.ihfft(a)
        result = cpfft.hfft(cp.array(freq), norm=norm)
        expected = np.fft.hfft(freq, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    @pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
    def test_ihfft_norm(self, norm):
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        result = cpfft.ihfft(cp.array(a), norm=norm)
        expected = np.fft.ihfft(a, norm=norm)
        npt.assert_allclose(result.get(), expected, **FFT_TOL)

    def test_fft_norm_default_matches_none(self):
        """Verify default (no norm arg) matches norm=None."""
        np.random.seed(42)
        a = np.random.randn(16).astype(np.float32)
        r1 = cpfft.fft(cp.array(a))
        r2 = cpfft.fft(cp.array(a), norm=None)
        npt.assert_array_equal(r1.get(), r2.get())


# ======================================================================
# Sorting parameter tests
# ======================================================================

class TestSortParams:
    """Test kind= and order= parameters on sort/argsort/partition/argpartition."""

    @pytest.mark.parametrize("kind", [None, "quicksort", "mergesort", "heapsort", "stable"])
    def test_sort_kind(self, kind):
        np.random.seed(42)
        a = np.random.randn(20).astype(np.float32)
        result = cp.sort(cp.array(a), kind=kind)
        expected = np.sort(a, kind=kind)
        npt.assert_array_equal(result.get(), expected)

    def test_sort_kind_default_unchanged(self):
        """Verify default (no kind) still works."""
        np.random.seed(42)
        a = np.random.randn(20).astype(np.float32)
        result = cp.sort(cp.array(a))
        expected = np.sort(a)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("kind", [None, "quicksort", "mergesort", "heapsort", "stable"])
    def test_argsort_kind(self, kind):
        np.random.seed(42)
        a = np.random.randn(20).astype(np.float32)
        result = cp.argsort(cp.array(a), kind=kind)
        expected = np.argsort(a, kind=kind)
        npt.assert_array_equal(result.get(), expected)

    def test_argsort_kind_default_unchanged(self):
        np.random.seed(42)
        a = np.random.randn(20).astype(np.float32)
        result = cp.argsort(cp.array(a))
        expected = np.argsort(a)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("kind", ["introselect"])
    def test_partition_kind(self, kind):
        np.random.seed(42)
        a = np.random.randn(20).astype(np.float32)
        result = cp.partition(cp.array(a), 5, kind=kind)
        expected = np.partition(a, 5, kind=kind)
        # Partition only guarantees kth element is in correct position
        npt.assert_allclose(result.get()[5], expected[5], rtol=1e-5)

    def test_partition_default_unchanged(self):
        np.random.seed(42)
        a = np.random.randn(20).astype(np.float32)
        result = cp.partition(cp.array(a), 5)
        expected = np.partition(a, 5)
        npt.assert_allclose(result.get()[5], expected[5], rtol=1e-5)

    @pytest.mark.parametrize("kind", ["introselect"])
    def test_argpartition_kind(self, kind):
        np.random.seed(42)
        a = np.random.randn(20).astype(np.float32)
        result = cp.argpartition(cp.array(a), 5, kind=kind)
        expected = np.argpartition(a, 5, kind=kind)
        # The kth position should hold index pointing to the same value
        npt.assert_allclose(a[result.get()[5]], a[expected[5]], rtol=1e-5)

    def test_argpartition_default_unchanged(self):
        np.random.seed(42)
        a = np.random.randn(20).astype(np.float32)
        result = cp.argpartition(cp.array(a), 5)
        expected = np.argpartition(a, 5)
        npt.assert_allclose(a[result.get()[5]], a[expected[5]], rtol=1e-5)

    def test_sort_order_param_accepted(self):
        """Verify order= param is accepted (None for non-structured arrays)."""
        np.random.seed(42)
        a = np.random.randn(20).astype(np.float32)
        # order=None is the default; just verify it's accepted without error
        result = cp.sort(cp.array(a), order=None)
        expected = np.sort(a, order=None)
        npt.assert_array_equal(result.get(), expected)

    def test_argsort_order_param_accepted(self):
        """Verify order= param is accepted (None for non-structured arrays)."""
        np.random.seed(42)
        a = np.random.randn(20).astype(np.float32)
        result = cp.argsort(cp.array(a), order=None)
        expected = np.argsort(a, order=None)
        npt.assert_array_equal(result.get(), expected)


class TestSearchsortedParams:
    """Test sorter= parameter on searchsorted."""

    def test_sorter(self):
        a = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        sorter = np.argsort(a)
        v = np.array([1.5, 2.5], dtype=np.float32)
        result = cp.searchsorted(cp.array(a), cp.array(v), sorter=sorter)
        expected = np.searchsorted(a, v, sorter=sorter)
        npt.assert_array_equal(result.get(), expected)

    def test_sorter_none_default(self):
        """Verify default (no sorter) still works."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = np.array([1.5, 2.5], dtype=np.float32)
        result = cp.searchsorted(cp.array(a), cp.array(v))
        expected = np.searchsorted(a, v)
        npt.assert_array_equal(result.get(), expected)

    def test_sorter_with_side(self):
        a = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        sorter = np.argsort(a)
        v = np.array([2.0], dtype=np.float32)
        result = cp.searchsorted(cp.array(a), cp.array(v), side='right', sorter=sorter)
        expected = np.searchsorted(a, v, side='right', sorter=sorter)
        npt.assert_array_equal(result.get(), expected)


# ======================================================================
# Indexing parameter tests
# ======================================================================

class TestTakeParams:
    """Test out= and mode= parameters on take."""

    @pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
    def test_take_mode(self, mode):
        a = np.arange(10, dtype=np.float32)
        if mode == "raise":
            indices = np.array([0, 3, 7])
        else:
            # Use out-of-range indices for wrap/clip
            indices = np.array([0, 3, 12, -2])
        result = cp.take(cp.array(a), indices, mode=mode)
        expected = np.take(a, indices, mode=mode)
        npt.assert_array_equal(result.get(), expected)

    def test_take_mode_default_is_raise(self):
        """Default mode should be 'raise'."""
        a = np.arange(10, dtype=np.float32)
        indices = np.array([0, 3, 7])
        r1 = cp.take(cp.array(a), indices)
        r2 = cp.take(cp.array(a), indices, mode='raise')
        npt.assert_array_equal(r1.get(), r2.get())

    def test_take_out(self):
        a = np.arange(10, dtype=np.float32)
        indices = np.array([0, 3, 7])
        out = cp.zeros(3, dtype=np.float32)
        result = cp.take(cp.array(a), indices, out=out)
        expected = np.take(a, indices)
        npt.assert_array_equal(out.get(), expected)
        # result should be the same object as out
        assert result is out

    def test_take_out_with_axis(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        indices = np.array([0, 2])
        out = cp.zeros((2, 4), dtype=np.float32)
        result = cp.take(cp.array(a), indices, axis=0, out=out)
        expected = np.take(a, indices, axis=0)
        npt.assert_array_equal(out.get(), expected)
        assert result is out


class TestChooseParams:
    """Test out= and mode= parameters on choose."""

    @pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
    def test_choose_mode(self, mode):
        a = np.array([0, 1, 2, 1], dtype=np.intp)
        choices = [
            np.array([0, 1, 2, 3], dtype=np.float32),
            np.array([10, 11, 12, 13], dtype=np.float32),
            np.array([20, 21, 22, 23], dtype=np.float32),
        ]
        if mode != "raise":
            # Use out-of-range index for wrap/clip
            a_mod = np.array([0, 1, 5, 1], dtype=np.intp)
        else:
            a_mod = a
        result = cp.choose(cp.array(a_mod), [cp.array(c) for c in choices], mode=mode)
        expected = np.choose(a_mod, choices, mode=mode)
        npt.assert_array_equal(result.get(), expected)

    def test_choose_out(self):
        a = np.array([0, 1, 2, 1], dtype=np.intp)
        choices = [
            np.array([0, 1, 2, 3], dtype=np.float32),
            np.array([10, 11, 12, 13], dtype=np.float32),
            np.array([20, 21, 22, 23], dtype=np.float32),
        ]
        out = cp.zeros(4, dtype=np.float32)
        result = cp.choose(cp.array(a), [cp.array(c) for c in choices], out=out)
        expected = np.choose(a, choices)
        npt.assert_array_equal(out.get(), expected)
        assert result is out

    def test_choose_default_mode_is_raise(self):
        a = np.array([0, 1, 2], dtype=np.intp)
        choices = [
            np.array([0, 1, 2], dtype=np.float32),
            np.array([10, 11, 12], dtype=np.float32),
            np.array([20, 21, 22], dtype=np.float32),
        ]
        r1 = cp.choose(cp.array(a), [cp.array(c) for c in choices])
        r2 = cp.choose(cp.array(a), [cp.array(c) for c in choices], mode='raise')
        npt.assert_array_equal(r1.get(), r2.get())


class TestCompressParams:
    """Test out= parameter on compress."""

    def test_compress_out(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        cond = np.array([True, False, True, False, True])
        out = cp.zeros(3, dtype=np.float32)
        result = cp.compress(cond, cp.array(a), out=out)
        expected = np.compress(cond, a)
        npt.assert_array_equal(out.get(), expected)
        assert result is out

    def test_compress_out_with_axis(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cond = np.array([True, False, True])
        out = cp.zeros((2, 4), dtype=np.float32)
        result = cp.compress(cond, cp.array(a), axis=0, out=out)
        expected = np.compress(cond, a, axis=0)
        npt.assert_array_equal(out.get(), expected)
        assert result is out

    def test_compress_default_no_out(self):
        """Verify default (no out) still works."""
        a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        cond = np.array([True, False, True, False, True])
        result = cp.compress(cond, cp.array(a))
        expected = np.compress(cond, a)
        npt.assert_array_equal(result.get(), expected)


class TestPutParams:
    """Test mode= parameter on put."""

    @pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
    def test_put_mode(self, mode):
        a_np = np.arange(5, dtype=np.float32)
        a_cp = cp.array(a_np.copy())
        if mode == "raise":
            ind = np.array([0, 2, 4])
        else:
            ind = np.array([0, 2, 7])  # out-of-range for wrap/clip
        v = np.array([10, 20, 30], dtype=np.float32)
        np.put(a_np, ind, v, mode=mode)
        cp.put(a_cp, ind, v, mode=mode)
        npt.assert_array_equal(a_cp.get(), a_np)

    def test_put_default_mode_is_raise(self):
        """Default mode should be 'raise'."""
        a_np = np.arange(5, dtype=np.float32)
        a_cp = cp.array(a_np.copy())
        ind = np.array([0, 2, 4])
        v = np.array([10, 20, 30], dtype=np.float32)
        np.put(a_np, ind, v)
        cp.put(a_cp, ind, v)
        npt.assert_array_equal(a_cp.get(), a_np)
