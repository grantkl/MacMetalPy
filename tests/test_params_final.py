"""Tests for parameter gap fixes in nan_ops, reductions, math_ops, indexing, linalg, linalg_top."""
import numpy as np
import pytest

import macmetalpy as mp


# ================================================================== nan_ops: histogram params

class TestHistogramParams:
    def test_histogram_range(self):
        a = mp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        hist, edges = mp.histogram(a, bins=5, range=(0.0, 10.0))
        hist_np, edges_np = np.histogram(a.get(), bins=5, range=(0.0, 10.0))
        np.testing.assert_array_equal(hist.get(), hist_np)
        np.testing.assert_array_almost_equal(edges.get(), edges_np)

    def test_histogram_density(self):
        a = mp.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        hist, edges = mp.histogram(a, bins=3, density=True)
        hist_np, edges_np = np.histogram(a.get(), bins=3, density=True)
        np.testing.assert_array_almost_equal(hist.get(), hist_np)
        np.testing.assert_array_almost_equal(edges.get(), edges_np)

    def test_histogram_weights(self):
        a = mp.array([1.0, 2.0, 3.0])
        w = mp.array([0.5, 1.0, 1.5])
        hist, edges = mp.histogram(a, bins=3, weights=w)
        hist_np, edges_np = np.histogram(a.get(), bins=3, weights=w.get())
        np.testing.assert_array_almost_equal(hist.get(), hist_np)
        np.testing.assert_array_almost_equal(edges.get(), edges_np)

    def test_histogram_range_density_weights(self):
        a = mp.array([1.0, 2.0, 3.0, 4.0])
        w = mp.array([1.0, 2.0, 1.0, 2.0])
        hist, edges = mp.histogram(a, bins=4, range=(0.0, 5.0), density=True, weights=w)
        hist_np, edges_np = np.histogram(a.get(), bins=4, range=(0.0, 5.0), density=True, weights=w.get())
        np.testing.assert_array_almost_equal(hist.get(), hist_np)
        np.testing.assert_array_almost_equal(edges.get(), edges_np)


class TestHistogram2dParams:
    def test_histogram2d_range(self):
        x = mp.array([1.0, 2.0, 3.0])
        y = mp.array([4.0, 5.0, 6.0])
        H, xe, ye = mp.histogram2d(x, y, bins=3, range=[[0.0, 4.0], [3.0, 7.0]])
        H_np, xe_np, ye_np = np.histogram2d(x.get(), y.get(), bins=3, range=[[0.0, 4.0], [3.0, 7.0]])
        np.testing.assert_array_equal(H.get(), H_np)
        np.testing.assert_array_almost_equal(xe.get(), xe_np)
        np.testing.assert_array_almost_equal(ye.get(), ye_np)

    def test_histogram2d_density(self):
        x = mp.array([1.0, 2.0, 3.0, 1.0])
        y = mp.array([4.0, 5.0, 6.0, 4.0])
        H, xe, ye = mp.histogram2d(x, y, bins=3, density=True)
        H_np, xe_np, ye_np = np.histogram2d(x.get(), y.get(), bins=3, density=True)
        np.testing.assert_array_almost_equal(H.get(), H_np)

    def test_histogram2d_weights(self):
        x = mp.array([1.0, 2.0, 3.0])
        y = mp.array([4.0, 5.0, 6.0])
        w = mp.array([0.5, 1.0, 1.5])
        H, xe, ye = mp.histogram2d(x, y, bins=3, weights=w)
        H_np, xe_np, ye_np = np.histogram2d(x.get(), y.get(), bins=3, weights=w.get())
        np.testing.assert_array_almost_equal(H.get(), H_np)


class TestHistogramddParams:
    def test_histogramdd_range(self):
        sample = mp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        H, edges = mp.histogramdd(sample, bins=3, range=[[0.0, 4.0], [3.0, 7.0]])
        H_np, edges_np = np.histogramdd(sample.get(), bins=3, range=[[0.0, 4.0], [3.0, 7.0]])
        np.testing.assert_array_equal(H.get(), H_np)

    def test_histogramdd_density(self):
        sample = mp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0], [1.0, 4.0]])
        H, edges = mp.histogramdd(sample, bins=3, density=True)
        H_np, edges_np = np.histogramdd(sample.get(), bins=3, density=True)
        np.testing.assert_array_almost_equal(H.get(), H_np)

    def test_histogramdd_weights(self):
        sample = mp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        w = mp.array([0.5, 1.0, 1.5])
        H, edges = mp.histogramdd(sample, bins=3, weights=w)
        H_np, edges_np = np.histogramdd(sample.get(), bins=3, weights=w.get())
        np.testing.assert_array_almost_equal(H.get(), H_np)


# ================================================================== reductions: diff params

class TestDiffParams:
    def test_diff_prepend(self):
        a = mp.array([1.0, 3.0, 6.0, 10.0])
        result = mp.diff(a, prepend=mp.array([0.0]))
        expected = np.diff(a.get(), prepend=np.array([0.0]))
        np.testing.assert_array_almost_equal(result.get(), expected)

    def test_diff_append(self):
        a = mp.array([1.0, 3.0, 6.0, 10.0])
        result = mp.diff(a, append=mp.array([15.0]))
        expected = np.diff(a.get(), append=np.array([15.0]))
        np.testing.assert_array_almost_equal(result.get(), expected)

    def test_diff_prepend_and_append(self):
        a = mp.array([1.0, 3.0, 6.0])
        result = mp.diff(a, prepend=mp.array([0.0]), append=mp.array([10.0]))
        expected = np.diff(a.get(), prepend=np.array([0.0]), append=np.array([10.0]))
        np.testing.assert_array_almost_equal(result.get(), expected)

    def test_diff_prepend_scalar(self):
        """prepend/append with plain numpy arrays (not ndarray) should also work."""
        a = mp.array([1.0, 3.0, 6.0])
        result = mp.diff(a, prepend=np.array([0.0]))
        expected = np.diff(a.get(), prepend=np.array([0.0]))
        np.testing.assert_array_almost_equal(result.get(), expected)


# ================================================================== math_ops: diagonal params

class TestDiagonalParams:
    def test_diagonal_axis1_axis2(self):
        a = mp.array(np.arange(24).reshape(2, 3, 4).astype(np.float32))
        result = mp.diagonal(a, offset=0, axis1=1, axis2=2)
        expected = np.diagonal(a.get(), offset=0, axis1=1, axis2=2)
        np.testing.assert_array_equal(result.get(), expected)

    def test_diagonal_different_axes(self):
        a = mp.array(np.arange(24).reshape(2, 3, 4).astype(np.float32))
        result = mp.diagonal(a, offset=0, axis1=0, axis2=2)
        expected = np.diagonal(a.get(), offset=0, axis1=0, axis2=2)
        np.testing.assert_array_equal(result.get(), expected)

    def test_diagonal_2d_default(self):
        a = mp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = mp.diagonal(a)
        expected = np.diagonal(a.get())
        np.testing.assert_array_equal(result.get(), expected)


# ================================================================== indexing: fill_diagonal params

class TestFillDiagonalParams:
    def test_fill_diagonal_wrap_false(self):
        a = mp.zeros((5, 3), dtype=np.float32)
        mp.fill_diagonal(a, 1.0, wrap=False)
        expected = np.zeros((5, 3), dtype=np.float32)
        np.fill_diagonal(expected, 1.0, wrap=False)
        np.testing.assert_array_equal(a.get(), expected)

    def test_fill_diagonal_wrap_true(self):
        a = mp.zeros((5, 3), dtype=np.float32)
        mp.fill_diagonal(a, 1.0, wrap=True)
        expected = np.zeros((5, 3), dtype=np.float32)
        np.fill_diagonal(expected, 1.0, wrap=True)
        np.testing.assert_array_equal(a.get(), expected)


# ================================================================== linalg: matrix_rank hermitian

class TestMatrixRankHermitian:
    def test_matrix_rank_hermitian(self):
        # Symmetric matrix
        a = mp.array([[1.0, 0.0], [0.0, 1.0]])
        result = mp.linalg.matrix_rank(a, hermitian=True)
        expected = np.linalg.matrix_rank(a.get(), hermitian=True)
        np.testing.assert_array_equal(result.get(), np.array(expected))

    def test_matrix_rank_hermitian_false(self):
        a = mp.array([[1.0, 2.0], [3.0, 4.0]])
        result = mp.linalg.matrix_rank(a, hermitian=False)
        expected = np.linalg.matrix_rank(a.get(), hermitian=False)
        np.testing.assert_array_equal(result.get(), np.array(expected))


# ================================================================== linalg: pinv hermitian

class TestPinvHermitian:
    def test_pinv_hermitian(self):
        a = mp.array([[1.0, 0.0], [0.0, 2.0]])
        result = mp.linalg.pinv(a, hermitian=True)
        expected = np.linalg.pinv(a.get(), hermitian=True)
        np.testing.assert_array_almost_equal(result.get(), expected)

    def test_pinv_hermitian_false(self):
        a = mp.array([[1.0, 2.0], [3.0, 4.0]])
        result = mp.linalg.pinv(a, hermitian=False)
        expected = np.linalg.pinv(a.get(), hermitian=False)
        np.testing.assert_array_almost_equal(result.get(), expected)


# ================================================================== linalg_top: einsum_path

class TestEinsumPath:
    def test_einsum_path_basic(self):
        a = mp.array(np.random.rand(5, 5).astype(np.float32))
        b = mp.array(np.random.rand(5, 5).astype(np.float32))
        from macmetalpy import linalg_top
        path, info = linalg_top.einsum_path('ij,jk->ik', a, b)
        path_np, info_np = np.einsum_path('ij,jk->ik', a.get(), b.get())
        assert path == path_np

    def test_einsum_path_optimize(self):
        a = mp.array(np.random.rand(3, 3).astype(np.float32))
        b = mp.array(np.random.rand(3, 3).astype(np.float32))
        c = mp.array(np.random.rand(3, 3).astype(np.float32))
        from macmetalpy import linalg_top
        path, info = linalg_top.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
        path_np, info_np = np.einsum_path('ij,jk,kl->il', a.get(), b.get(), c.get(), optimize='greedy')
        assert path == path_np


# ================================================================== reductions: divmod

class TestDivmod:
    def test_divmod_basic(self):
        from macmetalpy import reductions as red
        x1 = mp.array([7.0, 10.0, -3.0])
        x2 = mp.array([3.0, 4.0, 2.0])
        q, r = red.divmod(x1, x2)
        q_np, r_np = np.divmod(x1.get(), x2.get())
        np.testing.assert_array_almost_equal(q.get(), q_np)
        np.testing.assert_array_almost_equal(r.get(), r_np)


# ================================================================== bitwise_not alias

class TestBitwiseNot:
    def test_bitwise_not_exists(self):
        from macmetalpy import bitwise_ops
        assert hasattr(bitwise_ops, 'bitwise_not')
        a = mp.array([1, 0, -1])
        result = bitwise_ops.bitwise_not(a)
        expected = bitwise_ops.invert(a)
        np.testing.assert_array_equal(result.get(), expected.get())


# ================================================================== indexing: mask_indices

class TestMaskIndices:
    def test_mask_indices_triu(self):
        from macmetalpy import indexing
        rows, cols = indexing.mask_indices(3, np.triu)
        rows_np, cols_np = np.mask_indices(3, np.triu)
        np.testing.assert_array_equal(rows.get(), rows_np)
        np.testing.assert_array_equal(cols.get(), cols_np)

    def test_mask_indices_triu_k(self):
        from macmetalpy import indexing
        rows, cols = indexing.mask_indices(3, np.triu, k=1)
        rows_np, cols_np = np.mask_indices(3, np.triu, k=1)
        np.testing.assert_array_equal(rows.get(), rows_np)
        np.testing.assert_array_equal(cols.get(), cols_np)
