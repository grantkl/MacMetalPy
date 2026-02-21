"""Tests for random module gap-filling: aliases, new distributions, permuted, default_rng.

TDD: these tests are written FIRST, then the implementation in random.py.
"""

import numpy as np
import numpy.testing as npt
import pytest

from macmetalpy import random as cpr


N_SAMPLES = 100_000
MOMENT_TOL = dict(rtol=0.15, atol=0.15)


# ======================================================================
# Aliases: random_sample, ranf, sample
# ======================================================================

class TestRandomSample:
    def test_shape(self):
        result = cpr.random_sample(size=(3, 4))
        assert result.shape == (3, 4)

    def test_range(self):
        cpr.seed(0)
        vals = cpr.random_sample(size=1000).get()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0

    def test_dtype(self):
        result = cpr.random_sample(size=5)
        assert result.dtype == np.float32

    def test_scalar(self):
        result = cpr.random_sample()
        assert result.shape == ()


class TestRanf:
    def test_is_callable(self):
        """ranf should be the same function as random."""
        assert cpr.ranf is cpr.random


class TestSample:
    def test_is_callable(self):
        """sample should be the same function as random."""
        assert cpr.sample is cpr.random


# ======================================================================
# power distribution
# ======================================================================

class TestPower:
    def test_shape(self):
        result = cpr.power(5.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_range(self):
        cpr.seed(0)
        vals = cpr.power(2.0, size=1000).get()
        assert vals.min() >= 0.0
        assert vals.max() <= 1.0

    def test_dtype(self):
        result = cpr.power(3.0, size=10)
        assert result.dtype == np.float32

    def test_moments(self):
        cpr.seed(42)
        a = 5.0
        vals = cpr.power(a, size=N_SAMPLES).get().astype(np.float64)
        expected_mean = a / (a + 1.0)
        npt.assert_allclose(vals.mean(), expected_mean, **MOMENT_TOL)


# ======================================================================
# noncentral_chisquare
# ======================================================================

class TestNoncentralChisquare:
    def test_shape(self):
        result = cpr.noncentral_chisquare(3.0, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_positive(self):
        cpr.seed(0)
        vals = cpr.noncentral_chisquare(5.0, 2.0, size=1000).get()
        assert vals.min() >= 0.0

    def test_dtype(self):
        result = cpr.noncentral_chisquare(3.0, 1.0, size=10)
        assert result.dtype == np.float32

    def test_moments(self):
        cpr.seed(42)
        df, nonc = 5.0, 3.0
        vals = cpr.noncentral_chisquare(df, nonc, size=N_SAMPLES).get().astype(np.float64)
        expected_mean = df + nonc
        npt.assert_allclose(vals.mean(), expected_mean, **MOMENT_TOL)


# ======================================================================
# noncentral_f
# ======================================================================

class TestNoncentralF:
    def test_shape(self):
        result = cpr.noncentral_f(5, 10, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_positive(self):
        cpr.seed(0)
        vals = cpr.noncentral_f(5, 10, 2.0, size=1000).get()
        assert vals.min() >= 0.0

    def test_dtype(self):
        result = cpr.noncentral_f(5, 10, 1.0, size=10)
        assert result.dtype == np.float32


# ======================================================================
# permuted
# ======================================================================

class TestPermuted:
    def test_returns_new_array(self):
        """permuted should not modify the input array."""
        cpr.seed(0)
        a = cpr.rand(20)
        original = a.get().copy()
        result = cpr.permuted(a)
        # original array should be unchanged
        npt.assert_array_equal(a.get(), original)
        # result should contain the same elements
        assert sorted(result.get().tolist()) == sorted(original.tolist())

    def test_shape(self):
        cpr.seed(0)
        a = cpr.rand(5, 4)
        result = cpr.permuted(a)
        assert result.shape == a.shape

    def test_dtype(self):
        a = cpr.rand(10)
        result = cpr.permuted(a)
        assert result.dtype == a.dtype


# ======================================================================
# default_rng / Generator
# ======================================================================

class TestDefaultRng:
    def test_returns_generator(self):
        rng = cpr.default_rng(42)
        assert hasattr(rng, 'random')
        assert hasattr(rng, 'normal')
        assert hasattr(rng, 'integers')

    def test_random(self):
        rng = cpr.default_rng(42)
        result = rng.random(size=(3, 4))
        assert result.shape == (3, 4)
        assert result.dtype == np.float32

    def test_integers(self):
        rng = cpr.default_rng(42)
        result = rng.integers(0, 10, size=5)
        assert result.shape == (5,)
        vals = result.get()
        assert all(0 <= v < 10 for v in vals)

    def test_normal(self):
        rng = cpr.default_rng(42)
        result = rng.normal(0, 1, size=(4, 5))
        assert result.shape == (4, 5)
        assert result.dtype == np.float32

    def test_uniform(self):
        rng = cpr.default_rng(42)
        result = rng.uniform(0, 1, size=10)
        assert result.shape == (10,)

    def test_seed_reproducibility(self):
        rng1 = cpr.default_rng(123)
        a = rng1.random(size=100)
        rng2 = cpr.default_rng(123)
        b = rng2.random(size=100)
        npt.assert_array_equal(a.get(), b.get())

    def test_has_distribution_methods(self):
        """Generator should expose all common distribution methods."""
        rng = cpr.default_rng(0)
        expected_methods = [
            'random', 'integers', 'normal', 'uniform', 'standard_normal',
            'beta', 'binomial', 'chisquare', 'exponential', 'f', 'gamma',
            'geometric', 'gumbel', 'laplace', 'logistic', 'lognormal',
            'multinomial', 'multivariate_normal', 'negative_binomial',
            'pareto', 'permutation', 'permuted', 'poisson', 'power',
            'rayleigh', 'shuffle', 'standard_cauchy', 'standard_exponential',
            'standard_gamma', 'standard_t', 'triangular', 'vonmises',
            'wald', 'weibull', 'zipf', 'dirichlet', 'choice',
            'noncentral_chisquare', 'noncentral_f',
        ]
        for method in expected_methods:
            assert hasattr(rng, method), f"Generator missing method: {method}"
