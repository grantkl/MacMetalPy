"""Tests for kernel fusion of elementwise operations."""

import numpy as np
import pytest

import macmetalpy as mp


class TestFusionCorrectness:
    """Verify fused operations produce correct results vs NumPy."""

    def test_binary_chain_sin_cos_mul_add(self):
        """sin(a) + cos(b) * c — 4 ops fused into 1."""
        np.random.seed(42)
        a_np = np.random.randn(100_000).astype(np.float32)
        b_np = np.random.randn(100_000).astype(np.float32)
        c_np = np.random.randn(100_000).astype(np.float32)
        a, b, c = mp.array(a_np), mp.array(b_np), mp.array(c_np)
        result = mp.sin(a) + mp.cos(b) * c
        expected = np.sin(a_np) + np.cos(b_np) * c_np
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_unary_chain_sin_cos_exp(self):
        """sin(cos(exp(a))) — 3 nested unary ops."""
        np.random.seed(42)
        a_np = np.random.randn(100_000).astype(np.float32) * 0.5
        a = mp.array(a_np)
        result = mp.sin(mp.cos(mp.exp(a)))
        expected = np.sin(np.cos(np.exp(a_np)))
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_mixed_unary_binary(self):
        """exp(a) + log(b) — mix of unary and binary."""
        np.random.seed(42)
        a_np = np.abs(np.random.randn(100_000).astype(np.float32)) + 0.1
        b_np = np.abs(np.random.randn(100_000).astype(np.float32)) + 0.1
        a, b = mp.array(a_np), mp.array(b_np)
        result = mp.exp(a * 0.1) + mp.log(b)
        expected = np.exp(a_np * 0.1) + np.log(b_np)
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_sqrt_square_add(self):
        """sqrt(a) + square(b) — light ops."""
        np.random.seed(42)
        a_np = np.abs(np.random.randn(300_000).astype(np.float32)) + 0.01
        b_np = np.random.randn(300_000).astype(np.float32)
        a, b = mp.array(a_np), mp.array(b_np)
        result = mp.sqrt(a) + b * b
        expected = np.sqrt(a_np) + b_np * b_np
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_neg_abs_add(self):
        """neg + abs + add."""
        np.random.seed(42)
        a_np = np.random.randn(300_000).astype(np.float32)
        b_np = np.random.randn(300_000).astype(np.float32)
        a, b = mp.array(a_np), mp.array(b_np)
        result = (-a) + mp.abs(b)
        expected = (-a_np) + np.abs(b_np)
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)


class TestFusionMaterialization:
    """Verify fusion triggers materialize at the right points."""

    def test_get_materializes(self):
        """Calling .get() on lazy array produces correct result."""
        a_np = np.ones(100_000, dtype=np.float32)
        a = mp.array(a_np)
        result = mp.sin(a) + mp.cos(a)
        got = result.get()
        expected = np.sin(a_np) + np.cos(a_np)
        np.testing.assert_allclose(got, expected, rtol=1e-5)

    def test_reduction_materializes(self):
        """Calling .sum() on lazy array materializes first."""
        a_np = np.ones(100_000, dtype=np.float32) * 2.0
        a = mp.array(a_np)
        lazy = mp.sin(a)
        result = lazy.sum()
        expected = np.sin(a_np).sum()
        np.testing.assert_allclose(float(result), float(expected), rtol=1e-4)

    def test_getitem_materializes(self):
        """Indexing a lazy array materializes it."""
        a_np = np.arange(100_000, dtype=np.float32)
        a = mp.array(a_np)
        lazy = mp.exp(a * 0.0001)
        # Indexing should force materialization
        val = lazy[0]
        expected = np.exp(a_np[0] * 0.0001)
        np.testing.assert_allclose(float(val), float(expected), rtol=1e-5)

    def test_repr_materializes(self):
        """repr() of a lazy array should work (materializes)."""
        a_np = np.ones(100_000, dtype=np.float32)
        a = mp.array(a_np)
        lazy = mp.sin(a)
        r = repr(lazy)
        assert "array" in r.lower() or len(r) > 0


class TestFusionEdgeCases:
    """Edge cases for the fusion system."""

    def test_shared_subexpression(self):
        """t = sin(a); t + t — shared node should be computed once."""
        np.random.seed(42)
        a_np = np.random.randn(100_000).astype(np.float32)
        a = mp.array(a_np)
        t = mp.sin(a)
        result = t + t
        expected = np.sin(a_np) + np.sin(a_np)
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_max_depth_enforcement(self):
        """Chain of 20 ops should exceed MAX_DEPTH=16 and still produce correct result."""
        a_np = np.ones(100_000, dtype=np.float32) * 0.5
        a = mp.array(a_np)
        result = a
        expected = a_np.copy()
        for _ in range(20):
            result = result + result * 0.001
            expected = expected + expected * 0.001
        np.testing.assert_allclose(result.get(), expected, rtol=1e-4)

    def test_small_arrays_use_cpu(self):
        """Arrays below GPU threshold should NOT use fusion (CPU fast path)."""
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = mp.array(a_np)
        result = mp.sin(a) + mp.cos(a)
        expected = np.sin(a_np) + np.cos(a_np)
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_size_zero_array(self):
        """Empty arrays should not fuse."""
        a_np = np.array([], dtype=np.float32)
        a = mp.array(a_np)
        result = mp.sin(a)
        assert result.size == 0

    def test_inplace_ops_work(self):
        """In-place ops (+=) should work with fusion."""
        np.random.seed(42)
        a_np = np.random.randn(100_000).astype(np.float32)
        b_np = np.random.randn(100_000).astype(np.float32)
        a = mp.array(a_np.copy())
        b = mp.array(b_np)
        lazy_sin = mp.sin(b)
        a += lazy_sin
        expected = a_np + np.sin(b_np)
        np.testing.assert_allclose(a.get(), expected, rtol=1e-5)

    def test_float16_fusion(self):
        """Fusion should work with float16 dtype."""
        a_np = np.random.randn(100_000).astype(np.float16)
        b_np = np.random.randn(100_000).astype(np.float16)
        a, b = mp.array(a_np), mp.array(b_np)
        result = a + b
        expected = a_np + b_np
        np.testing.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-3)


class TestFusionOps:
    """Test specific fuseable operations for correctness."""

    def test_tanh_fusion(self):
        np.random.seed(42)
        a_np = np.random.randn(100_000).astype(np.float32) * 5
        a = mp.array(a_np)
        result = mp.tanh(mp.sin(a))
        expected = np.tanh(np.sin(a_np))
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_floor_ceil(self):
        a_np = np.random.randn(100_000).astype(np.float32) * 10
        a = mp.array(a_np)
        result = mp.floor(a) + mp.ceil(a)
        expected = np.floor(a_np) + np.ceil(a_np)
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_log_exp_roundtrip(self):
        a_np = np.abs(np.random.randn(100_000).astype(np.float32)) + 0.1
        a = mp.array(a_np)
        result = mp.log(mp.exp(a))
        expected = a_np  # log(exp(x)) = x
        np.testing.assert_allclose(result.get(), expected, rtol=1e-4, atol=1e-5)

    def test_binary_op_in_chain(self):
        """Test binary op fusion when chained with a unary op."""
        np.random.seed(42)
        a_np = np.random.randn(100_000).astype(np.float32)
        b_np = np.abs(np.random.randn(100_000).astype(np.float32)) + 0.1
        a, b = mp.array(a_np), mp.array(b_np)
        # sin(a) makes a lazy, then add triggers fusion
        result = mp.sin(a) + b
        expected = np.sin(a_np) + b_np
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_multiply_in_chain(self):
        """Test multiply fusion when chained with unary ops."""
        a_np = np.random.randn(100_000).astype(np.float32)
        b_np = np.random.randn(100_000).astype(np.float32)
        a, b = mp.array(a_np), mp.array(b_np)
        # sin(cos(a)) builds a 2-deep lazy chain, then * sin(b) fuses
        result = mp.sin(mp.cos(a)) * mp.sin(b)
        expected = np.sin(np.cos(a_np)) * np.sin(b_np)
        np.testing.assert_allclose(result.get(), expected, rtol=1e-5)
