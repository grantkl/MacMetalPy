"""Tests for untested random APIs.

Tests verify shape, dtype, and basic statistical properties.
Uses seeding for reproducibility where applicable.
macmetalpy uses float32 (no float64).
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
import macmetalpy.random as cpr


N_SAMPLES = 50_000
MOMENT_TOL = dict(rtol=0.15, atol=0.15)


# ======================================================================
# beta
# ======================================================================

class TestBeta:
    def test_shape(self):
        result = cpr.beta(2.0, 5.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.beta(2.0, 5.0, size=10)
        assert result.dtype == np.float32

    def test_range_and_mean(self):
        cpr.seed(42)
        vals = cpr.beta(2.0, 5.0, size=N_SAMPLES).get()
        assert vals.min() >= 0.0
        assert vals.max() <= 1.0
        expected_mean = 2.0 / (2.0 + 5.0)
        npt.assert_allclose(vals.mean(), expected_mean, **MOMENT_TOL)


# ======================================================================
# binomial
# ======================================================================

class TestBinomial:
    def test_shape(self):
        result = cpr.binomial(10, 0.5, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.binomial(10, 0.5, size=10)
        assert result.dtype == np.int32

    def test_range_and_mean(self):
        cpr.seed(42)
        vals = cpr.binomial(10, 0.5, size=N_SAMPLES).get()
        assert vals.min() >= 0
        assert vals.max() <= 10
        npt.assert_allclose(vals.mean(), 10 * 0.5, **MOMENT_TOL)


# ======================================================================
# chisquare
# ======================================================================

class TestChisquare:
    def test_shape(self):
        result = cpr.chisquare(3.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.chisquare(3.0, size=10)
        assert result.dtype == np.float32

    def test_positive_and_mean(self):
        cpr.seed(42)
        vals = cpr.chisquare(5.0, size=N_SAMPLES).get()
        assert vals.min() >= 0.0
        npt.assert_allclose(vals.mean(), 5.0, **MOMENT_TOL)


# ======================================================================
# default_rng
# ======================================================================

class TestDefaultRng:
    def test_returns_generator(self):
        rng = cpr.default_rng(42)
        assert hasattr(rng, 'random')
        assert hasattr(rng, 'normal')

    def test_random(self):
        rng = cpr.default_rng(42)
        result = rng.random(size=(3, 4))
        assert result.shape == (3, 4)
        assert result.dtype == np.float32

    def test_seed_reproducibility(self):
        rng1 = cpr.default_rng(123)
        a = rng1.random(size=100)
        rng2 = cpr.default_rng(123)
        b = rng2.random(size=100)
        npt.assert_array_equal(a.get(), b.get())


# ======================================================================
# dirichlet
# ======================================================================

class TestDirichlet:
    def test_shape(self):
        result = cpr.dirichlet([1.0, 2.0, 3.0], size=5)
        assert result.shape == (5, 3)

    def test_dtype(self):
        result = cpr.dirichlet([1.0, 2.0], size=10)
        assert result.dtype == np.float32

    def test_sums_to_one(self):
        cpr.seed(42)
        vals = cpr.dirichlet([1.0, 2.0, 3.0], size=100).get()
        npt.assert_allclose(vals.sum(axis=1), 1.0, rtol=1e-3, atol=1e-3)


# ======================================================================
# exponential
# ======================================================================

class TestExponential:
    def test_shape(self):
        result = cpr.exponential(1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.exponential(1.0, size=10)
        assert result.dtype == np.float32

    def test_positive_and_mean(self):
        cpr.seed(42)
        scale = 3.0
        vals = cpr.exponential(scale, size=N_SAMPLES).get()
        assert vals.min() >= 0.0
        npt.assert_allclose(vals.mean(), scale, **MOMENT_TOL)


# ======================================================================
# f
# ======================================================================

class TestF:
    def test_shape(self):
        result = cpr.f(5, 10, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.f(5, 10, size=10)
        assert result.dtype == np.float32

    def test_positive_and_mean(self):
        cpr.seed(42)
        d2 = 10.0
        vals = cpr.f(5, d2, size=N_SAMPLES).get()
        assert vals.min() >= 0.0
        # For F(d1,d2), mean = d2/(d2-2) when d2 > 2
        expected_mean = d2 / (d2 - 2.0)
        npt.assert_allclose(vals.mean(), expected_mean, **MOMENT_TOL)


# ======================================================================
# gamma
# ======================================================================

class TestGamma:
    def test_shape(self):
        result = cpr.gamma(2.0, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.gamma(2.0, 1.0, size=10)
        assert result.dtype == np.float32

    def test_positive_and_mean(self):
        cpr.seed(42)
        k, theta = 2.0, 3.0
        vals = cpr.gamma(k, theta, size=N_SAMPLES).get()
        assert vals.min() >= 0.0
        npt.assert_allclose(vals.mean(), k * theta, **MOMENT_TOL)


# ======================================================================
# geometric
# ======================================================================

class TestGeometric:
    def test_shape(self):
        result = cpr.geometric(0.5, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.geometric(0.5, size=10)
        assert result.dtype == np.int32

    def test_positive_and_mean(self):
        cpr.seed(42)
        p = 0.3
        vals = cpr.geometric(p, size=N_SAMPLES).get().astype(np.float64)
        assert vals.min() >= 1
        # geometric mean = 1/p
        npt.assert_allclose(vals.mean(), 1.0 / p, **MOMENT_TOL)


# ======================================================================
# gumbel
# ======================================================================

class TestGumbel:
    def test_shape(self):
        result = cpr.gumbel(0.0, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.gumbel(0.0, 1.0, size=10)
        assert result.dtype == np.float32

    def test_mean(self):
        cpr.seed(42)
        mu, beta = 0.0, 1.0
        vals = cpr.gumbel(mu, beta, size=N_SAMPLES).get()
        # Gumbel mean = mu + beta * euler_gamma
        euler_gamma = 0.5772156649
        npt.assert_allclose(vals.mean(), mu + beta * euler_gamma, **MOMENT_TOL)


# ======================================================================
# hypergeometric
# ======================================================================

class TestHypergeometric:
    def test_shape(self):
        result = cpr.hypergeometric(10, 5, 7, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.hypergeometric(10, 5, 7, size=10)
        assert result.dtype == np.int32

    def test_range_and_mean(self):
        cpr.seed(42)
        ngood, nbad, nsample = 10, 5, 7
        vals = cpr.hypergeometric(ngood, nbad, nsample, size=N_SAMPLES).get()
        assert vals.min() >= 0
        assert vals.max() <= nsample
        # mean = nsample * ngood / (ngood + nbad)
        expected_mean = nsample * ngood / (ngood + nbad)
        npt.assert_allclose(vals.mean(), expected_mean, **MOMENT_TOL)


# ======================================================================
# laplace
# ======================================================================

class TestLaplace:
    def test_shape(self):
        result = cpr.laplace(0.0, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.laplace(0.0, 1.0, size=10)
        assert result.dtype == np.float32

    def test_mean(self):
        cpr.seed(42)
        loc = 2.0
        vals = cpr.laplace(loc, 1.0, size=N_SAMPLES).get()
        npt.assert_allclose(vals.mean(), loc, **MOMENT_TOL)


# ======================================================================
# logistic
# ======================================================================

class TestLogistic:
    def test_shape(self):
        result = cpr.logistic(0.0, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.logistic(0.0, 1.0, size=10)
        assert result.dtype == np.float32

    def test_mean(self):
        cpr.seed(42)
        loc = 3.0
        vals = cpr.logistic(loc, 1.0, size=N_SAMPLES).get()
        npt.assert_allclose(vals.mean(), loc, **MOMENT_TOL)


# ======================================================================
# lognormal
# ======================================================================

class TestLognormal:
    def test_shape(self):
        result = cpr.lognormal(0.0, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.lognormal(0.0, 1.0, size=10)
        assert result.dtype == np.float32

    def test_positive(self):
        cpr.seed(42)
        vals = cpr.lognormal(0.0, 1.0, size=N_SAMPLES).get()
        assert vals.min() > 0.0


# ======================================================================
# logseries
# ======================================================================

class TestLogseries:
    def test_shape(self):
        result = cpr.logseries(0.5, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.logseries(0.5, size=10)
        assert result.dtype == np.int32

    def test_positive(self):
        cpr.seed(42)
        vals = cpr.logseries(0.5, size=N_SAMPLES).get()
        assert vals.min() >= 1


# ======================================================================
# multinomial
# ======================================================================

class TestMultinomial:
    def test_shape(self):
        result = cpr.multinomial(10, [0.2, 0.3, 0.5], size=5)
        assert result.shape == (5, 3)

    def test_dtype(self):
        result = cpr.multinomial(10, [0.5, 0.5], size=10)
        assert result.dtype == np.int32

    def test_sums(self):
        cpr.seed(42)
        n = 20
        vals = cpr.multinomial(n, [0.2, 0.3, 0.5], size=100).get()
        npt.assert_array_equal(vals.sum(axis=1), n)


# ======================================================================
# multivariate_normal
# ======================================================================

class TestMultivariateNormal:
    def test_shape(self):
        mean = [0.0, 0.0]
        cov = [[1.0, 0.0], [0.0, 1.0]]
        result = cpr.multivariate_normal(mean, cov, size=5)
        assert result.shape == (5, 2)

    def test_dtype(self):
        mean = [0.0, 0.0]
        cov = [[1.0, 0.0], [0.0, 1.0]]
        result = cpr.multivariate_normal(mean, cov, size=10)
        assert result.dtype == np.float32

    def test_mean(self):
        cpr.seed(42)
        mean = [1.0, 2.0]
        cov = [[1.0, 0.0], [0.0, 1.0]]
        vals = cpr.multivariate_normal(mean, cov, size=N_SAMPLES).get()
        npt.assert_allclose(vals.mean(axis=0), mean, **MOMENT_TOL)


# ======================================================================
# negative_binomial
# ======================================================================

class TestNegativeBinomial:
    def test_shape(self):
        result = cpr.negative_binomial(5, 0.5, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.negative_binomial(5, 0.5, size=10)
        assert result.dtype == np.int32

    def test_non_negative_and_mean(self):
        cpr.seed(42)
        n, p = 5, 0.5
        vals = cpr.negative_binomial(n, p, size=N_SAMPLES).get().astype(np.float64)
        assert vals.min() >= 0
        # mean = n*(1-p)/p
        npt.assert_allclose(vals.mean(), n * (1 - p) / p, **MOMENT_TOL)


# ======================================================================
# noncentral_chisquare
# ======================================================================

class TestNoncentralChisquare:
    def test_shape(self):
        result = cpr.noncentral_chisquare(3.0, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.noncentral_chisquare(3.0, 1.0, size=10)
        assert result.dtype == np.float32

    def test_positive_and_mean(self):
        cpr.seed(42)
        df, nonc = 5.0, 3.0
        vals = cpr.noncentral_chisquare(df, nonc, size=N_SAMPLES).get()
        assert vals.min() >= 0.0
        # mean = df + nonc
        npt.assert_allclose(vals.mean(), df + nonc, **MOMENT_TOL)


# ======================================================================
# noncentral_f
# ======================================================================

class TestNoncentralF:
    def test_shape(self):
        result = cpr.noncentral_f(5, 10, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.noncentral_f(5, 10, 1.0, size=10)
        assert result.dtype == np.float32

    def test_positive(self):
        cpr.seed(42)
        vals = cpr.noncentral_f(5, 10, 2.0, size=N_SAMPLES).get()
        assert vals.min() >= 0.0


# ======================================================================
# pareto
# ======================================================================

class TestPareto:
    def test_shape(self):
        result = cpr.pareto(3.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.pareto(3.0, size=10)
        assert result.dtype == np.float32

    def test_non_negative(self):
        cpr.seed(42)
        vals = cpr.pareto(3.0, size=N_SAMPLES).get()
        assert vals.min() >= 0.0


# ======================================================================
# permuted
# ======================================================================

class TestPermuted:
    def test_shape(self):
        cpr.seed(42)
        a = cpr.rand(5, 4)
        result = cpr.permuted(a)
        assert result.shape == a.shape

    def test_dtype(self):
        a = cpr.rand(10)
        result = cpr.permuted(a)
        assert result.dtype == a.dtype

    def test_does_not_modify_input(self):
        cpr.seed(0)
        a = cpr.rand(20)
        original = a.get().copy()
        cpr.permuted(a)
        npt.assert_array_equal(a.get(), original)


# ======================================================================
# poisson
# ======================================================================

class TestPoisson:
    def test_shape(self):
        result = cpr.poisson(5.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.poisson(5.0, size=10)
        assert result.dtype == np.int32

    def test_non_negative_and_mean(self):
        cpr.seed(42)
        lam = 7.0
        vals = cpr.poisson(lam, size=N_SAMPLES).get().astype(np.float64)
        assert vals.min() >= 0
        npt.assert_allclose(vals.mean(), lam, **MOMENT_TOL)


# ======================================================================
# randint
# ======================================================================

class TestRandint:
    def test_shape(self):
        result = cpr.randint(0, 10, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.randint(0, 10, size=10)
        assert result.dtype == np.int32

    def test_range(self):
        cpr.seed(42)
        vals = cpr.randint(5, 15, size=N_SAMPLES).get()
        assert vals.min() >= 5
        assert vals.max() < 15


# ======================================================================
# randn
# ======================================================================

class TestRandn:
    def test_shape(self):
        result = cpr.randn(3, 4)
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.randn(10)
        assert result.dtype == np.float32

    def test_moments(self):
        cpr.seed(42)
        vals = cpr.randn(N_SAMPLES).get()
        npt.assert_allclose(vals.mean(), 0.0, atol=0.05)
        npt.assert_allclose(vals.var(), 1.0, **MOMENT_TOL)


# ======================================================================
# random_sample
# ======================================================================

class TestRandomSample:
    def test_shape(self):
        result = cpr.random_sample(size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.random_sample(size=10)
        assert result.dtype == np.float32

    def test_range(self):
        cpr.seed(42)
        vals = cpr.random_sample(size=N_SAMPLES).get()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0


# ======================================================================
# ranf
# ======================================================================

class TestRanf:
    def test_is_alias(self):
        assert cpr.ranf is cpr.random


# ======================================================================
# rayleigh
# ======================================================================

class TestRayleigh:
    def test_shape(self):
        result = cpr.rayleigh(1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.rayleigh(1.0, size=10)
        assert result.dtype == np.float32

    def test_positive_and_mean(self):
        cpr.seed(42)
        sigma = 2.0
        vals = cpr.rayleigh(sigma, size=N_SAMPLES).get()
        assert vals.min() >= 0.0
        # Rayleigh mean = sigma * sqrt(pi/2)
        expected = sigma * np.sqrt(np.pi / 2.0)
        npt.assert_allclose(vals.mean(), expected, **MOMENT_TOL)


# ======================================================================
# sample
# ======================================================================

class TestSample:
    def test_is_alias(self):
        assert cpr.sample is cpr.random


# ======================================================================
# standard_cauchy
# ======================================================================

class TestStandardCauchy:
    def test_shape(self):
        result = cpr.standard_cauchy(size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.standard_cauchy(size=10)
        assert result.dtype == np.float32


# ======================================================================
# standard_exponential
# ======================================================================

class TestStandardExponential:
    def test_shape(self):
        result = cpr.standard_exponential(size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.standard_exponential(size=10)
        assert result.dtype == np.float32

    def test_positive_and_mean(self):
        cpr.seed(42)
        vals = cpr.standard_exponential(size=N_SAMPLES).get()
        assert vals.min() >= 0.0
        npt.assert_allclose(vals.mean(), 1.0, **MOMENT_TOL)


# ======================================================================
# standard_gamma
# ======================================================================

class TestStandardGamma:
    def test_shape(self):
        result = cpr.standard_gamma(2.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.standard_gamma(2.0, size=10)
        assert result.dtype == np.float32

    def test_positive_and_mean(self):
        cpr.seed(42)
        shape_param = 3.0
        vals = cpr.standard_gamma(shape_param, size=N_SAMPLES).get()
        assert vals.min() >= 0.0
        # standard_gamma(k) has mean = k
        npt.assert_allclose(vals.mean(), shape_param, **MOMENT_TOL)


# ======================================================================
# standard_normal
# ======================================================================

class TestStandardNormal:
    def test_shape(self):
        result = cpr.standard_normal(size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.standard_normal(size=10)
        assert result.dtype == np.float32

    def test_moments(self):
        cpr.seed(42)
        vals = cpr.standard_normal(size=N_SAMPLES).get()
        npt.assert_allclose(vals.mean(), 0.0, atol=0.05)
        npt.assert_allclose(vals.var(), 1.0, **MOMENT_TOL)


# ======================================================================
# standard_t
# ======================================================================

class TestStandardT:
    def test_shape(self):
        result = cpr.standard_t(5.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.standard_t(5.0, size=10)
        assert result.dtype == np.float32

    def test_mean(self):
        cpr.seed(42)
        # For df > 1, mean = 0
        vals = cpr.standard_t(5.0, size=N_SAMPLES).get()
        npt.assert_allclose(vals.mean(), 0.0, atol=0.1)


# ======================================================================
# triangular
# ======================================================================

class TestTriangular:
    def test_shape(self):
        result = cpr.triangular(0.0, 0.5, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.triangular(0.0, 0.5, 1.0, size=10)
        assert result.dtype == np.float32

    def test_range_and_mean(self):
        cpr.seed(42)
        left, mode, right = 0.0, 0.5, 1.0
        vals = cpr.triangular(left, mode, right, size=N_SAMPLES).get()
        assert vals.min() >= left
        assert vals.max() <= right
        expected_mean = (left + mode + right) / 3.0
        npt.assert_allclose(vals.mean(), expected_mean, **MOMENT_TOL)


# ======================================================================
# vonmises
# ======================================================================

class TestVonmises:
    def test_shape(self):
        result = cpr.vonmises(0.0, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.vonmises(0.0, 1.0, size=10)
        assert result.dtype == np.float32

    def test_range(self):
        cpr.seed(42)
        vals = cpr.vonmises(0.0, 1.0, size=N_SAMPLES).get()
        assert vals.min() >= -np.pi
        assert vals.max() <= np.pi


# ======================================================================
# wald
# ======================================================================

class TestWald:
    def test_shape(self):
        result = cpr.wald(1.0, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.wald(1.0, 1.0, size=10)
        assert result.dtype == np.float32

    def test_positive_and_mean(self):
        cpr.seed(42)
        mean_param, scale = 3.0, 5.0
        vals = cpr.wald(mean_param, scale, size=N_SAMPLES).get()
        assert vals.min() > 0.0
        # Wald mean = mean_param
        npt.assert_allclose(vals.mean(), mean_param, **MOMENT_TOL)


# ======================================================================
# weibull
# ======================================================================

class TestWeibull:
    def test_shape(self):
        result = cpr.weibull(2.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.weibull(2.0, size=10)
        assert result.dtype == np.float32

    def test_non_negative(self):
        cpr.seed(42)
        vals = cpr.weibull(2.0, size=N_SAMPLES).get()
        assert vals.min() >= 0.0


# ======================================================================
# zipf
# ======================================================================

class TestZipf:
    def test_shape(self):
        result = cpr.zipf(2.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.zipf(2.0, size=10)
        assert result.dtype == np.int32

    def test_positive(self):
        cpr.seed(42)
        vals = cpr.zipf(2.0, size=N_SAMPLES).get()
        assert vals.min() >= 1


# ======================================================================
# Generator
# ======================================================================

class TestGenerator:
    def test_create_generator_with_seed(self):
        gen = cp.random.Generator(42)
        assert gen is not None
        assert hasattr(gen, 'random')
        assert hasattr(gen, 'integers')
        assert hasattr(gen, 'normal')
        assert hasattr(gen, 'uniform')

    def test_create_generator_no_seed(self):
        gen = cp.random.Generator()
        result = gen.random(size=5)
        assert result.shape == (5,)

    def test_random_shape_and_dtype(self):
        gen = cp.random.Generator(42)
        result = gen.random(size=(3, 4))
        assert result.shape == (3, 4)
        assert result.dtype == np.float32

    def test_random_values_in_range(self):
        gen = cp.random.Generator(42)
        vals = gen.random(size=1000).get()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0

    def test_integers_shape_and_range(self):
        gen = cp.random.Generator(42)
        result = gen.integers(0, 10, size=5)
        assert result.shape == (5,)
        vals = result.get()
        assert vals.min() >= 0
        assert vals.max() < 10

    def test_integers_dtype(self):
        gen = cp.random.Generator(42)
        result = gen.integers(0, 10, size=5)
        assert result.dtype == np.int32

    def test_normal_shape(self):
        gen = cp.random.Generator(42)
        result = gen.normal(0, 1, size=100)
        assert result.shape == (100,)

    def test_normal_approximate_mean_std(self):
        gen = cp.random.Generator(42)
        vals = gen.normal(0, 1, size=N_SAMPLES).get()
        npt.assert_allclose(vals.mean(), 0.0, atol=0.1)
        npt.assert_allclose(vals.std(), 1.0, **MOMENT_TOL)

    def test_uniform_shape_and_range(self):
        gen = cp.random.Generator(42)
        result = gen.uniform(0, 1, size=50)
        assert result.shape == (50,)
        vals = result.get()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0

    def test_uniform_dtype(self):
        gen = cp.random.Generator(42)
        result = gen.uniform(0, 1, size=50)
        assert result.dtype == np.float32

    def test_seed_reproducibility(self):
        gen1 = cp.random.Generator(123)
        a = gen1.random(size=100)
        gen2 = cp.random.Generator(123)
        b = gen2.random(size=100)
        npt.assert_array_equal(a.get(), b.get())
