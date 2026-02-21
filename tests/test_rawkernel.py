"""Tests for RawKernel (user-defined Metal kernels).

Ref: cupy_tests/core_tests/test_raw.py
Target: ~16 cases.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy import RawKernel


# ── kernel sources ────────────────────────────────────────────────────────

FLOAT32_DOUBLE = '''
#include <metal_stdlib>
using namespace metal;
kernel void dbl(device float *a [[buffer(0)]],
                device float *out [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * 2.0f;
}
'''

INT32_DOUBLE = '''
#include <metal_stdlib>
using namespace metal;
kernel void dbl_int(device int *a [[buffer(0)]],
                    device int *out [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * 2;
}
'''

FLOAT16_DOUBLE = '''
#include <metal_stdlib>
using namespace metal;
kernel void dbl_half(device half *a [[buffer(0)]],
                     device half *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * half(2.0);
}
'''

UINT32_DOUBLE = '''
#include <metal_stdlib>
using namespace metal;
kernel void dbl_uint(device uint *a [[buffer(0)]],
                     device uint *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * 2u;
}
'''

ADD_KERNEL = '''
#include <metal_stdlib>
using namespace metal;
kernel void add_k(device float *a [[buffer(0)]],
                  device float *b [[buffer(1)]],
                  device float *out [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}
'''

EIGHT_BUFFER_KERNEL = '''
#include <metal_stdlib>
using namespace metal;
kernel void sum8(device float *a [[buffer(0)]],
                 device float *b [[buffer(1)]],
                 device float *c [[buffer(2)]],
                 device float *d [[buffer(3)]],
                 device float *e [[buffer(4)]],
                 device float *f [[buffer(5)]],
                 device float *g [[buffer(6)]],
                 device float *h [[buffer(7)]],
                 device float *out [[buffer(8)]],
                 uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id] + c[id] + d[id] + e[id] + f[id] + g[id] + h[id];
}
'''

SYNC_EXPLICIT = '''
#include <metal_stdlib>
using namespace metal;
kernel void dbl_sync(device float *a [[buffer(0)]],
                     device float *out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * 2.0f;
}
kernel void _sync() {}
'''


# ======================================================================
# Tests
# ======================================================================

class TestRawKernelBasic:
    def test_float32(self):
        k = RawKernel(FLOAT32_DOUBLE, 'dbl')
        a = cp.array([1, 2, 3], dtype=np.float32)
        out = cp.empty(3, dtype=np.float32)
        k(3, (a, out))
        npt.assert_allclose(out.get(), [2, 4, 6])

    def test_int32(self):
        k = RawKernel(INT32_DOUBLE, 'dbl_int')
        a = cp.array([1, 2, 3], dtype=np.int32)
        out = cp.empty(3, dtype=np.int32)
        k(3, (a, out))
        npt.assert_array_equal(out.get(), [2, 4, 6])

    def test_float16(self):
        k = RawKernel(FLOAT16_DOUBLE, 'dbl_half')
        a = cp.array([1, 2, 3], dtype=np.float16)
        out = cp.empty(3, dtype=np.float16)
        k(3, (a, out))
        npt.assert_allclose(out.get(), [2, 4, 6], rtol=1e-2)

    def test_uint32(self):
        k = RawKernel(UINT32_DOUBLE, 'dbl_uint')
        a = cp.array([1, 2, 3], dtype=np.uint32)
        out = cp.empty(3, dtype=np.uint32)
        k(3, (a, out))
        npt.assert_array_equal(out.get(), [2, 4, 6])


class TestRawKernelGrid:
    def test_int_grid(self):
        k = RawKernel(FLOAT32_DOUBLE, 'dbl')
        a = cp.array([1, 2, 3, 4], dtype=np.float32)
        out = cp.empty(4, dtype=np.float32)
        k(4, (a, out))
        npt.assert_allclose(out.get(), [2, 4, 6, 8])

    def test_1_tuple(self):
        k = RawKernel(FLOAT32_DOUBLE, 'dbl')
        a = cp.array([1, 2, 3], dtype=np.float32)
        out = cp.empty(3, dtype=np.float32)
        k((3,), (a, out))
        npt.assert_allclose(out.get(), [2, 4, 6])

    def test_2_tuple(self):
        k = RawKernel(FLOAT32_DOUBLE, 'dbl')
        a = cp.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        out = cp.empty(6, dtype=np.float32)
        k((6,), (a, out))
        npt.assert_allclose(out.get(), [2, 4, 6, 8, 10, 12])

    def test_3_tuple(self):
        k = RawKernel(FLOAT32_DOUBLE, 'dbl')
        a = cp.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        out = cp.empty(6, dtype=np.float32)
        k((6, 1, 1), (a, out))
        npt.assert_allclose(out.get(), [2, 4, 6, 8, 10, 12])

    def test_4_tuple_raises(self):
        k = RawKernel(FLOAT32_DOUBLE, 'dbl')
        a = cp.array([1], dtype=np.float32)
        out = cp.empty(1, dtype=np.float32)
        with pytest.raises(ValueError, match="1-3"):
            k((1, 1, 1, 1), (a, out))


class TestRawKernelSync:
    def test_auto_append_sync(self):
        # Source without _sync should still work (auto-appended)
        k = RawKernel(FLOAT32_DOUBLE, 'dbl')
        a = cp.array([10, 20], dtype=np.float32)
        out = cp.empty(2, dtype=np.float32)
        k(2, (a, out))
        npt.assert_allclose(out.get(), [20, 40])

    def test_source_already_has_sync(self):
        k = RawKernel(SYNC_EXPLICIT, 'dbl_sync')
        a = cp.array([10, 20], dtype=np.float32)
        out = cp.empty(2, dtype=np.float32)
        k(2, (a, out))
        npt.assert_allclose(out.get(), [20, 40])


class TestRawKernelAdvanced:
    def test_eight_buffers(self):
        k = RawKernel(EIGHT_BUFFER_KERNEL, 'sum8')
        bufs = [cp.array([float(i + 1)], dtype=np.float32) for i in range(8)]
        out = cp.empty(1, dtype=np.float32)
        k(1, (*bufs, out))
        expected = sum(range(1, 9))
        npt.assert_allclose(out.get(), [expected])

    def test_reuse_kernel(self):
        k = RawKernel(FLOAT32_DOUBLE, 'dbl')
        for val in [1, 5, 10]:
            a = cp.array([val], dtype=np.float32)
            out = cp.empty(1, dtype=np.float32)
            k(1, (a, out))
            npt.assert_allclose(out.get(), [val * 2])

    def test_two_different_kernels(self):
        k1 = RawKernel(FLOAT32_DOUBLE, 'dbl')
        k2 = RawKernel(ADD_KERNEL, 'add_k')
        a = cp.array([1, 2, 3], dtype=np.float32)
        b = cp.array([4, 5, 6], dtype=np.float32)
        out1 = cp.empty(3, dtype=np.float32)
        out2 = cp.empty(3, dtype=np.float32)
        k1(3, (a, out1))
        k2(3, (a, b, out2))
        npt.assert_allclose(out1.get(), [2, 4, 6])
        npt.assert_allclose(out2.get(), [5, 7, 9])

    def test_large_grid(self):
        n = 100_000
        k = RawKernel(FLOAT32_DOUBLE, 'dbl')
        a = cp.array(np.ones(n, dtype=np.float32))
        out = cp.empty(n, dtype=np.float32)
        k(n, (a, out))
        npt.assert_allclose(out.get(), np.full(n, 2.0))
