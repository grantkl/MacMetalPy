"""Advanced benchmark functions for macmetalpy vs NumPy.

Covers: linalg, FFT, random, manipulation, sorting, indexing, set ops,
NaN ops, polynomial, and I/O operations.
"""

import time
import tempfile
import os

import numpy as np

SIZE_MAP = {"small": 1_000, "medium": 100_000, "large": 1_000_000}


def _median_time(func, warmup=1, repeats=5):
    """Run func repeatedly and return median elapsed time."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    times.sort()
    return times[len(times) // 2]


# ============================================================================
# Linalg benchmarks (10)
# ============================================================================

def bench_matmul(size):
    import macmetalpy as cp
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)

    def mp_fn():
        r = cp.matmul(a_cp, b_cp)
        _ = r.get()

    def np_fn():
        _ = np.matmul(a_np, b_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_dot(size):
    import macmetalpy as cp
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)

    def mp_fn():
        r = cp.dot(a_cp, b_cp)
        _ = r.get()

    def np_fn():
        _ = np.dot(a_np, b_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_inv(size):
    import macmetalpy as cp
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_np = a_np @ a_np.T + n * np.eye(n, dtype=np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.linalg.inv(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.linalg.inv(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_solve(size):
    import macmetalpy as cp
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_np = a_np @ a_np.T + n * np.eye(n, dtype=np.float32)
    b_np = np.random.rand(n).astype(np.float32)
    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)

    def mp_fn():
        r = cp.linalg.solve(a_cp, b_cp)
        _ = r.get()

    def np_fn():
        _ = np.linalg.solve(a_np, b_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_eig(size):
    import macmetalpy as cp
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        w, v = cp.linalg.eig(a_cp)
        _ = w.get()

    def np_fn():
        _ = np.linalg.eig(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_svd(size):
    import macmetalpy as cp
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        u, s, vh = cp.linalg.svd(a_cp)
        _ = s.get()

    def np_fn():
        _ = np.linalg.svd(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_det(size):
    import macmetalpy as cp
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.linalg.det(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.linalg.det(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_norm(size):
    import macmetalpy as cp
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.linalg.norm(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.linalg.norm(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_cholesky(size):
    import macmetalpy as cp
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_np = a_np @ a_np.T + n * np.eye(n, dtype=np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.linalg.cholesky(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.linalg.cholesky(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_qr(size):
    import macmetalpy as cp
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        q, r = cp.linalg.qr(a_cp)
        _ = q.get()

    def np_fn():
        _ = np.linalg.qr(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


# ============================================================================
# FFT benchmarks (6)
# ============================================================================

def bench_fft(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.fft.fft(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.fft.fft(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_ifft(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.complex64)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.fft.ifft(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.fft.ifft(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_rfft(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.fft.rfft(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.fft.rfft(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_irfft(size):
    import macmetalpy as cp
    a_np = np.random.rand(size // 2 + 1).astype(np.complex64)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.fft.irfft(a_cp, n=size)
        _ = r.get()

    def np_fn():
        _ = np.fft.irfft(a_np, n=size)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_fft2(size):
    import macmetalpy as cp
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.fft.fft2(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.fft.fft2(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_ifft2(size):
    import macmetalpy as cp
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.complex64)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.fft.ifft2(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.fft.ifft2(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


# ============================================================================
# Random benchmarks (5)
# ============================================================================

def bench_rand(size):
    import macmetalpy as cp

    def mp_fn():
        r = cp.random.rand(size)
        _ = r.get()

    def np_fn():
        _ = np.random.rand(size)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_randn(size):
    import macmetalpy as cp

    def mp_fn():
        r = cp.random.randn(size)
        _ = r.get()

    def np_fn():
        _ = np.random.randn(size)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_randint(size):
    import macmetalpy as cp

    def mp_fn():
        r = cp.random.randint(0, 100, size=(size,))
        _ = r.get()

    def np_fn():
        _ = np.random.randint(0, 100, size=(size,))

    return _median_time(mp_fn), _median_time(np_fn)


def bench_uniform(size):
    import macmetalpy as cp

    def mp_fn():
        r = cp.random.uniform(0.0, 1.0, size=(size,))
        _ = r.get()

    def np_fn():
        _ = np.random.uniform(0.0, 1.0, size=(size,))

    return _median_time(mp_fn), _median_time(np_fn)


def bench_normal(size):
    import macmetalpy as cp

    def mp_fn():
        r = cp.random.normal(0.0, 1.0, size=(size,))
        _ = r.get()

    def np_fn():
        _ = np.random.normal(0.0, 1.0, size=(size,))

    return _median_time(mp_fn), _median_time(np_fn)


# ============================================================================
# Manipulation benchmarks (11)
# ============================================================================

def bench_reshape(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    new_shape = (10, size // 10)

    def mp_fn():
        r = cp.reshape(a_cp, new_shape)
        _ = r.get()

    def np_fn():
        _ = np.reshape(a_np, new_shape)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_transpose(size):
    import macmetalpy as cp
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.transpose(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.transpose(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_concatenate(size):
    import macmetalpy as cp
    half = size // 2
    a_np = np.random.rand(half).astype(np.float32)
    b_np = np.random.rand(half).astype(np.float32)
    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)

    def mp_fn():
        r = cp.concatenate([a_cp, b_cp])
        _ = r.get()

    def np_fn():
        _ = np.concatenate([a_np, b_np])

    return _median_time(mp_fn), _median_time(np_fn)


def bench_stack(size):
    import macmetalpy as cp
    half = size // 2
    a_np = np.random.rand(half).astype(np.float32)
    b_np = np.random.rand(half).astype(np.float32)
    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)

    def mp_fn():
        r = cp.stack([a_cp, b_cp])
        _ = r.get()

    def np_fn():
        _ = np.stack([a_np, b_np])

    return _median_time(mp_fn), _median_time(np_fn)


def bench_split(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        parts = cp.split(a_cp, 10)
        _ = parts[0].get()

    def np_fn():
        _ = np.split(a_np, 10)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_tile(size):
    import macmetalpy as cp
    a_np = np.random.rand(size // 10).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.tile(a_cp, 10)
        _ = r.get()

    def np_fn():
        _ = np.tile(a_np, 10)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_repeat(size):
    import macmetalpy as cp
    a_np = np.random.rand(size // 10).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.repeat(a_cp, 10)
        _ = r.get()

    def np_fn():
        _ = np.repeat(a_np, 10)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_flip(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.flip(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.flip(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_roll(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.roll(a_cp, size // 4)
        _ = r.get()

    def np_fn():
        _ = np.roll(a_np, size // 4)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_squeeze(size):
    import macmetalpy as cp
    a_np = np.random.rand(1, size, 1).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.squeeze(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.squeeze(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_ravel(size):
    import macmetalpy as cp
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.ravel(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.ravel(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


# ============================================================================
# Sorting benchmarks (5)
# ============================================================================

def bench_sort(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.sort(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.sort(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_argsort(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.argsort(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.argsort(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_unique(size):
    import macmetalpy as cp
    a_np = np.random.randint(0, size // 2, size=size).astype(np.int32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.unique(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.unique(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_searchsorted(size):
    import macmetalpy as cp
    a_np = np.sort(np.random.rand(size).astype(np.float32))
    v_np = np.random.rand(size // 10).astype(np.float32)
    a_cp = cp.array(a_np)
    v_cp = cp.array(v_np)

    def mp_fn():
        r = cp.searchsorted(a_cp, v_cp)
        _ = r.get()

    def np_fn():
        _ = np.searchsorted(a_np, v_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_partition(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    kth = size // 2

    def mp_fn():
        r = cp.partition(a_cp, kth)
        _ = r.get()

    def np_fn():
        _ = np.partition(a_np, kth)

    return _median_time(mp_fn), _median_time(np_fn)


# ============================================================================
# Indexing benchmarks (5)
# ============================================================================

def bench_take(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    idx_np = np.random.randint(0, size, size=size // 10).astype(np.int32)
    a_cp = cp.array(a_np)
    idx_cp = cp.array(idx_np)

    def mp_fn():
        r = cp.take(a_cp, idx_cp)
        _ = r.get()

    def np_fn():
        _ = np.take(a_np, idx_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_where(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    cond_np = a_np > 0.5
    a_cp = cp.array(a_np)
    cond_cp = cp.array(cond_np)
    b_cp = cp.array(np.zeros(size, dtype=np.float32))
    b_np = np.zeros(size, dtype=np.float32)

    def mp_fn():
        r = cp.where(cond_cp, a_cp, b_cp)
        _ = r.get()

    def np_fn():
        _ = np.where(cond_np, a_np, b_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_nonzero(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_np[a_np < 0.5] = 0.0
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.nonzero(a_cp)
        _ = r[0].get()

    def np_fn():
        _ = np.nonzero(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_argwhere(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_np[a_np < 0.5] = 0.0
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.argwhere(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.argwhere(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_put(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    idx_np = np.random.randint(0, size, size=size // 10).astype(np.int32)
    vals_np = np.random.rand(size // 10).astype(np.float32)
    a_cp = cp.array(a_np.copy())
    idx_cp = cp.array(idx_np)
    vals_cp = cp.array(vals_np)

    def mp_fn():
        cp.put(a_cp, idx_cp, vals_cp)
        _ = a_cp.get()

    def np_fn():
        a_tmp = a_np.copy()
        np.put(a_tmp, idx_np, vals_np)

    return _median_time(mp_fn), _median_time(np_fn)


# ============================================================================
# Set ops benchmarks (5)
# ============================================================================

def bench_set_unique(size):
    import macmetalpy as cp
    a_np = np.random.randint(0, size // 2, size=size).astype(np.int32)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.unique(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.unique(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_intersect1d(size):
    import macmetalpy as cp
    a_np = np.random.randint(0, size, size=size).astype(np.int32)
    b_np = np.random.randint(0, size, size=size).astype(np.int32)
    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)

    def mp_fn():
        r = cp.intersect1d(a_cp, b_cp)
        _ = r.get()

    def np_fn():
        _ = np.intersect1d(a_np, b_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_union1d(size):
    import macmetalpy as cp
    a_np = np.random.randint(0, size, size=size).astype(np.int32)
    b_np = np.random.randint(0, size, size=size).astype(np.int32)
    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)

    def mp_fn():
        r = cp.union1d(a_cp, b_cp)
        _ = r.get()

    def np_fn():
        _ = np.union1d(a_np, b_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_setdiff1d(size):
    import macmetalpy as cp
    a_np = np.random.randint(0, size, size=size).astype(np.int32)
    b_np = np.random.randint(0, size, size=size).astype(np.int32)
    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)

    def mp_fn():
        r = cp.setdiff1d(a_cp, b_cp)
        _ = r.get()

    def np_fn():
        _ = np.setdiff1d(a_np, b_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_isin(size):
    import macmetalpy as cp
    a_np = np.random.randint(0, size, size=size).astype(np.int32)
    b_np = np.random.randint(0, size, size=size // 10).astype(np.int32)
    a_cp = cp.array(a_np)
    b_cp = cp.array(b_np)

    def mp_fn():
        r = cp.isin(a_cp, b_cp)
        _ = r.get()

    def np_fn():
        _ = np.isin(a_np, b_np)

    return _median_time(mp_fn), _median_time(np_fn)


# ============================================================================
# NaN ops benchmarks (5)
# ============================================================================

def _make_nan_array(size):
    a = np.random.rand(size).astype(np.float32)
    mask = np.random.rand(size) < 0.1
    a[mask] = np.nan
    return a


def bench_nansum(size):
    import macmetalpy as cp
    a_np = _make_nan_array(size)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.nansum(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.nansum(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_nanmean(size):
    import macmetalpy as cp
    a_np = _make_nan_array(size)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.nanmean(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.nanmean(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_nanmax(size):
    import macmetalpy as cp
    a_np = _make_nan_array(size)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.nanmax(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.nanmax(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_nanmin(size):
    import macmetalpy as cp
    a_np = _make_nan_array(size)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.nanmin(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.nanmin(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_nanstd(size):
    import macmetalpy as cp
    a_np = _make_nan_array(size)
    a_cp = cp.array(a_np)

    def mp_fn():
        r = cp.nanstd(a_cp)
        _ = r.get()

    def np_fn():
        _ = np.nanstd(a_np)

    return _median_time(mp_fn), _median_time(np_fn)


# ============================================================================
# Polynomial benchmarks (3)
# ============================================================================

def bench_polyval(size):
    import macmetalpy as cp
    coeffs_np = np.random.rand(11).astype(np.float64)  # degree-10
    x_np = np.random.rand(size).astype(np.float64)
    coeffs_cp = cp.array(coeffs_np)
    x_cp = cp.array(x_np)

    def mp_fn():
        r = cp.polyval(coeffs_cp, x_cp)
        _ = r.get()

    def np_fn():
        _ = np.polyval(coeffs_np, x_np)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_polyfit(size):
    import macmetalpy as cp
    x_np = np.linspace(0, 1, size).astype(np.float64)
    y_np = np.random.rand(size).astype(np.float64)
    x_cp = cp.array(x_np)
    y_cp = cp.array(y_np)

    def mp_fn():
        r = cp.polyfit(x_cp, y_cp, 5)
        _ = r.get()

    def np_fn():
        _ = np.polyfit(x_np, y_np, 5)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_roots(size):
    import macmetalpy as cp
    # Cap degree to avoid extremely slow computation
    degree = min(size, 100)
    coeffs_np = np.random.rand(degree + 1).astype(np.float64)
    coeffs_np[0] = 1.0  # ensure leading coefficient is nonzero
    coeffs_cp = cp.array(coeffs_np)

    def mp_fn():
        r = cp.roots(coeffs_cp)
        _ = r.get()

    def np_fn():
        _ = np.roots(coeffs_np)

    return _median_time(mp_fn), _median_time(np_fn)


# ============================================================================
# IO benchmarks (2)
# ============================================================================

def bench_save_load(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            fname = f.name
        try:
            cp.save(fname, a_cp)
            r = cp.load(fname)
            _ = r.get()
        finally:
            os.unlink(fname)

    def np_fn():
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            fname = f.name
        try:
            np.save(fname, a_np)
            _ = np.load(fname)
        finally:
            os.unlink(fname)

    return _median_time(mp_fn), _median_time(np_fn)


def bench_loadtxt_savetxt(size):
    import macmetalpy as cp
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)

    def mp_fn():
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            fname = f.name
        try:
            cp.savetxt(fname, a_cp)
            r = cp.loadtxt(fname)
            _ = r.get()
        finally:
            os.unlink(fname)

    def np_fn():
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            fname = f.name
        try:
            np.savetxt(fname, a_np)
            _ = np.loadtxt(fname)
        finally:
            os.unlink(fname)

    return _median_time(mp_fn), _median_time(np_fn)


# ============================================================================
# BENCHMARKS registry
# ============================================================================

BENCHMARKS = [
    # --- Linalg (10) - only small and medium ---
    {"name": "matmul", "category": "linalg", "func": bench_matmul, "sizes": ["small", "medium"]},
    {"name": "dot", "category": "linalg", "func": bench_dot, "sizes": ["small", "medium"]},
    {"name": "inv", "category": "linalg", "func": bench_inv, "sizes": ["small", "medium"]},
    {"name": "solve", "category": "linalg", "func": bench_solve, "sizes": ["small", "medium"]},
    {"name": "eig", "category": "linalg", "func": bench_eig, "sizes": ["small", "medium"]},
    {"name": "svd", "category": "linalg", "func": bench_svd, "sizes": ["small", "medium"]},
    {"name": "det", "category": "linalg", "func": bench_det, "sizes": ["small", "medium"]},
    {"name": "norm", "category": "linalg", "func": bench_norm, "sizes": ["small", "medium"]},
    {"name": "cholesky", "category": "linalg", "func": bench_cholesky, "sizes": ["small", "medium"]},
    {"name": "qr", "category": "linalg", "func": bench_qr, "sizes": ["small", "medium"]},

    # --- FFT (6) ---
    {"name": "fft", "category": "fft", "func": bench_fft, "sizes": ["small", "medium", "large"]},
    {"name": "ifft", "category": "fft", "func": bench_ifft, "sizes": ["small", "medium", "large"]},
    {"name": "rfft", "category": "fft", "func": bench_rfft, "sizes": ["small", "medium", "large"]},
    {"name": "irfft", "category": "fft", "func": bench_irfft, "sizes": ["small", "medium", "large"]},
    {"name": "fft2", "category": "fft", "func": bench_fft2, "sizes": ["small", "medium", "large"]},
    {"name": "ifft2", "category": "fft", "func": bench_ifft2, "sizes": ["small", "medium", "large"]},

    # --- Random (5) ---
    {"name": "rand", "category": "random", "func": bench_rand, "sizes": ["small", "medium", "large"]},
    {"name": "randn", "category": "random", "func": bench_randn, "sizes": ["small", "medium", "large"]},
    {"name": "randint", "category": "random", "func": bench_randint, "sizes": ["small", "medium", "large"]},
    {"name": "uniform", "category": "random", "func": bench_uniform, "sizes": ["small", "medium", "large"]},
    {"name": "normal", "category": "random", "func": bench_normal, "sizes": ["small", "medium", "large"]},

    # --- Manipulation (11) ---
    {"name": "reshape", "category": "manipulation", "func": bench_reshape, "sizes": ["small", "medium", "large"]},
    {"name": "transpose", "category": "manipulation", "func": bench_transpose, "sizes": ["small", "medium", "large"]},
    {"name": "concatenate", "category": "manipulation", "func": bench_concatenate, "sizes": ["small", "medium", "large"]},
    {"name": "stack", "category": "manipulation", "func": bench_stack, "sizes": ["small", "medium", "large"]},
    {"name": "split", "category": "manipulation", "func": bench_split, "sizes": ["small", "medium", "large"]},
    {"name": "tile", "category": "manipulation", "func": bench_tile, "sizes": ["small", "medium", "large"]},
    {"name": "repeat", "category": "manipulation", "func": bench_repeat, "sizes": ["small", "medium", "large"]},
    {"name": "flip", "category": "manipulation", "func": bench_flip, "sizes": ["small", "medium", "large"]},
    {"name": "roll", "category": "manipulation", "func": bench_roll, "sizes": ["small", "medium", "large"]},
    {"name": "squeeze", "category": "manipulation", "func": bench_squeeze, "sizes": ["small", "medium", "large"]},
    {"name": "ravel", "category": "manipulation", "func": bench_ravel, "sizes": ["small", "medium", "large"]},

    # --- Sorting (5) ---
    {"name": "sort", "category": "sorting", "func": bench_sort, "sizes": ["small", "medium", "large"]},
    {"name": "argsort", "category": "sorting", "func": bench_argsort, "sizes": ["small", "medium", "large"]},
    {"name": "unique", "category": "sorting", "func": bench_unique, "sizes": ["small", "medium", "large"]},
    {"name": "searchsorted", "category": "sorting", "func": bench_searchsorted, "sizes": ["small", "medium", "large"]},
    {"name": "partition", "category": "sorting", "func": bench_partition, "sizes": ["small", "medium", "large"]},

    # --- Indexing (5) ---
    {"name": "take", "category": "indexing", "func": bench_take, "sizes": ["small", "medium", "large"]},
    {"name": "where", "category": "indexing", "func": bench_where, "sizes": ["small", "medium", "large"]},
    {"name": "nonzero", "category": "indexing", "func": bench_nonzero, "sizes": ["small", "medium", "large"]},
    {"name": "argwhere", "category": "indexing", "func": bench_argwhere, "sizes": ["small", "medium", "large"]},
    {"name": "put", "category": "indexing", "func": bench_put, "sizes": ["small", "medium", "large"]},

    # --- Set ops (5) ---
    {"name": "unique", "category": "set_ops", "func": bench_set_unique, "sizes": ["small", "medium", "large"]},
    {"name": "intersect1d", "category": "set_ops", "func": bench_intersect1d, "sizes": ["small", "medium", "large"]},
    {"name": "union1d", "category": "set_ops", "func": bench_union1d, "sizes": ["small", "medium", "large"]},
    {"name": "setdiff1d", "category": "set_ops", "func": bench_setdiff1d, "sizes": ["small", "medium", "large"]},
    {"name": "isin", "category": "set_ops", "func": bench_isin, "sizes": ["small", "medium", "large"]},

    # --- NaN ops (5) ---
    {"name": "nansum", "category": "nan_ops", "func": bench_nansum, "sizes": ["small", "medium", "large"]},
    {"name": "nanmean", "category": "nan_ops", "func": bench_nanmean, "sizes": ["small", "medium", "large"]},
    {"name": "nanmax", "category": "nan_ops", "func": bench_nanmax, "sizes": ["small", "medium", "large"]},
    {"name": "nanmin", "category": "nan_ops", "func": bench_nanmin, "sizes": ["small", "medium", "large"]},
    {"name": "nanstd", "category": "nan_ops", "func": bench_nanstd, "sizes": ["small", "medium", "large"]},

    # --- Poly (3) ---
    {"name": "polyval", "category": "poly", "func": bench_polyval, "sizes": ["small", "medium", "large"]},
    {"name": "polyfit", "category": "poly", "func": bench_polyfit, "sizes": ["small", "medium", "large"]},
    {"name": "roots", "category": "poly", "func": bench_roots, "sizes": ["small"]},

    # --- IO (2) - only small and medium ---
    {"name": "save_load", "category": "io", "func": bench_save_load, "sizes": ["small", "medium"]},
    {"name": "loadtxt_savetxt", "category": "io", "func": bench_loadtxt_savetxt, "sizes": ["small", "medium"]},
]
