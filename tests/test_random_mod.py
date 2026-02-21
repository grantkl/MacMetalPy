"""Tests for macmetalpy.random module."""
from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from macmetalpy import random as cpr


# ---------------------------------------------------------------------------
# seed / reproducibility
# ---------------------------------------------------------------------------

class TestSeed:
    def test_reproducible(self):
        cpr.seed(42)
        a = cpr.rand(10)
        cpr.seed(42)
        b = cpr.rand(10)
        assert_array_equal(a.get(), b.get())


# ---------------------------------------------------------------------------
# rand
# ---------------------------------------------------------------------------

class TestRand:
    def test_shape(self):
        result = cpr.rand(3, 4)
        assert result.shape == (3, 4)

    def test_range(self):
        cpr.seed(0)
        result = cpr.rand(1000)
        assert result.get().min() >= 0.0
        assert result.get().max() < 1.0

    def test_dtype(self):
        result = cpr.rand(5)
        assert result.dtype == np.float32



# ---------------------------------------------------------------------------
# randn
# ---------------------------------------------------------------------------

class TestRandn:
    def test_shape(self):
        result = cpr.randn(2, 3)
        assert result.shape == (2, 3)

    def test_dtype(self):
        result = cpr.randn(100)
        assert result.dtype == np.float32



# ---------------------------------------------------------------------------
# randint
# ---------------------------------------------------------------------------

class TestRandint:
    def test_shape(self):
        result = cpr.randint(0, 10, size=(3, 4))
        assert result.shape == (3, 4)

    def test_range(self):
        cpr.seed(0)
        result = cpr.randint(5, 15, size=1000)
        vals = result.get()
        assert vals.min() >= 5
        assert vals.max() < 15

    def test_dtype(self):
        result = cpr.randint(0, 10, size=5)
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# random
# ---------------------------------------------------------------------------

class TestRandom:
    def test_shape(self):
        result = cpr.random(size=(2, 5))
        assert result.shape == (2, 5)

    def test_range(self):
        cpr.seed(0)
        result = cpr.random(size=1000)
        vals = result.get()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0

    def test_dtype(self):
        result = cpr.random(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# shuffle
# ---------------------------------------------------------------------------

class TestShuffle:
    def test_modifies_in_place(self):
        cpr.seed(0)
        a = cpr.rand(10)
        original = a.get().copy()
        cpr.shuffle(a)
        # Very unlikely to stay in same order
        assert not np.array_equal(original, a.get()) or len(original) <= 1


# ---------------------------------------------------------------------------
# permutation
# ---------------------------------------------------------------------------

class TestPermutation:
    def test_int_input(self):
        cpr.seed(0)
        result = cpr.permutation(10)
        assert result.shape == (10,)
        assert sorted(result.get().tolist()) == list(range(10))

    def test_dtype_int(self):
        result = cpr.permutation(5)
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# choice
# ---------------------------------------------------------------------------

class TestChoice:
    def test_from_int(self):
        cpr.seed(0)
        result = cpr.choice(10, size=5)
        assert result.shape == (5,)
        vals = result.get()
        assert all(0 <= v < 10 for v in vals)

    def test_no_replace(self):
        cpr.seed(0)
        result = cpr.choice(10, size=10, replace=False)
        assert len(set(result.get().tolist())) == 10


# ---------------------------------------------------------------------------
# normal
# ---------------------------------------------------------------------------

class TestNormal:
    def test_shape(self):
        result = cpr.normal(0, 1, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.normal(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# uniform
# ---------------------------------------------------------------------------

class TestUniform:
    def test_shape(self):
        result = cpr.uniform(0, 1, size=(4, 5))
        assert result.shape == (4, 5)

    def test_range(self):
        cpr.seed(0)
        result = cpr.uniform(2.0, 5.0, size=1000)
        vals = result.get()
        assert vals.min() >= 2.0
        assert vals.max() <= 5.0

    def test_dtype(self):
        result = cpr.uniform(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------

class TestBeta:
    def test_shape(self):
        result = cpr.beta(2, 5, size=(3, 3))
        assert result.shape == (3, 3)

    def test_range(self):
        cpr.seed(0)
        result = cpr.beta(2, 5, size=1000)
        vals = result.get()
        assert vals.min() >= 0.0
        assert vals.max() <= 1.0

    def test_dtype(self):
        result = cpr.beta(1, 1, size=5)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# binomial
# ---------------------------------------------------------------------------

class TestBinomial:
    def test_shape(self):
        result = cpr.binomial(10, 0.5, size=(2, 3))
        assert result.shape == (2, 3)

    def test_range(self):
        cpr.seed(0)
        result = cpr.binomial(10, 0.5, size=1000)
        vals = result.get()
        assert vals.min() >= 0
        assert vals.max() <= 10

    def test_dtype(self):
        result = cpr.binomial(10, 0.5, size=5)
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# exponential
# ---------------------------------------------------------------------------

class TestExponential:
    def test_shape(self):
        result = cpr.exponential(1.0, size=(4, 4))
        assert result.shape == (4, 4)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.exponential(1.0, size=1000)
        assert result.get().min() >= 0.0

    def test_dtype(self):
        result = cpr.exponential(size=5)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# gamma
# ---------------------------------------------------------------------------

class TestGamma:
    def test_shape(self):
        result = cpr.gamma(2.0, 1.0, size=(3, 3))
        assert result.shape == (3, 3)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.gamma(2.0, 1.0, size=1000)
        assert result.get().min() >= 0.0

    def test_dtype(self):
        result = cpr.gamma(2.0, size=5)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# poisson
# ---------------------------------------------------------------------------

class TestPoisson:
    def test_shape(self):
        result = cpr.poisson(5.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_non_negative(self):
        cpr.seed(0)
        result = cpr.poisson(5.0, size=1000)
        assert result.get().min() >= 0

    def test_dtype(self):
        result = cpr.poisson(5.0, size=5)
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# standard_normal
# ---------------------------------------------------------------------------

class TestStandardNormal:
    def test_shape(self):
        result = cpr.standard_normal(size=(5, 5))
        assert result.shape == (5, 5)

    def test_dtype(self):
        result = cpr.standard_normal(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# standard_cauchy
# ---------------------------------------------------------------------------

class TestStandardCauchy:
    def test_shape(self):
        result = cpr.standard_cauchy(size=(3, 3))
        assert result.shape == (3, 3)

    def test_dtype(self):
        result = cpr.standard_cauchy(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# standard_exponential
# ---------------------------------------------------------------------------

class TestStandardExponential:
    def test_shape(self):
        result = cpr.standard_exponential(size=(4, 4))
        assert result.shape == (4, 4)

    def test_dtype(self):
        result = cpr.standard_exponential(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# standard_gamma
# ---------------------------------------------------------------------------

class TestStandardGamma:
    def test_shape(self):
        result = cpr.standard_gamma(2.0, size=(3, 3))
        assert result.shape == (3, 3)

    def test_dtype(self):
        result = cpr.standard_gamma(2.0, size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# standard_t
# ---------------------------------------------------------------------------

class TestStandardT:
    def test_shape(self):
        result = cpr.standard_t(5.0, size=(3, 3))
        assert result.shape == (3, 3)

    def test_dtype(self):
        result = cpr.standard_t(5.0, size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# chisquare
# ---------------------------------------------------------------------------

class TestChisquare:
    def test_shape(self):
        result = cpr.chisquare(3.0, size=(4, 4))
        assert result.shape == (4, 4)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.chisquare(3.0, size=1000)
        assert result.get().min() >= 0.0

    def test_dtype(self):
        result = cpr.chisquare(3.0, size=5)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# geometric
# ---------------------------------------------------------------------------

class TestGeometric:
    def test_shape(self):
        result = cpr.geometric(0.5, size=(3, 3))
        assert result.shape == (3, 3)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.geometric(0.5, size=1000)
        assert result.get().min() >= 1

    def test_dtype(self):
        result = cpr.geometric(0.5, size=5)
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# laplace
# ---------------------------------------------------------------------------

class TestLaplace:
    def test_shape(self):
        result = cpr.laplace(0, 1, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.laplace(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# logistic
# ---------------------------------------------------------------------------

class TestLogistic:
    def test_shape(self):
        result = cpr.logistic(0, 1, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.logistic(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# lognormal
# ---------------------------------------------------------------------------

class TestLognormal:
    def test_shape(self):
        result = cpr.lognormal(0, 1, size=(3, 4))
        assert result.shape == (3, 4)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.lognormal(0, 1, size=1000)
        assert result.get().min() > 0.0

    def test_dtype(self):
        result = cpr.lognormal(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# gumbel
# ---------------------------------------------------------------------------

class TestGumbel:
    def test_shape(self):
        result = cpr.gumbel(0, 1, size=(3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        result = cpr.gumbel(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# rayleigh
# ---------------------------------------------------------------------------

class TestRayleigh:
    def test_shape(self):
        result = cpr.rayleigh(1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.rayleigh(1.0, size=1000)
        assert result.get().min() >= 0.0

    def test_dtype(self):
        result = cpr.rayleigh(size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# triangular
# ---------------------------------------------------------------------------

class TestTriangular:
    def test_shape(self):
        result = cpr.triangular(0, 0.5, 1.0, size=(3, 4))
        assert result.shape == (3, 4)

    def test_range(self):
        cpr.seed(0)
        result = cpr.triangular(1.0, 3.0, 5.0, size=1000)
        vals = result.get()
        assert vals.min() >= 1.0
        assert vals.max() <= 5.0

    def test_dtype(self):
        result = cpr.triangular(0, 0.5, 1.0, size=5)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# weibull
# ---------------------------------------------------------------------------

class TestWeibull:
    def test_shape(self):
        result = cpr.weibull(2.0, size=(3, 3))
        assert result.shape == (3, 3)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.weibull(2.0, size=1000)
        assert result.get().min() >= 0.0

    def test_dtype(self):
        result = cpr.weibull(2.0, size=5)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# vonmises
# ---------------------------------------------------------------------------

class TestVonmises:
    def test_shape(self):
        result = cpr.vonmises(0.0, 1.0, size=(3, 3))
        assert result.shape == (3, 3)

    def test_dtype(self):
        result = cpr.vonmises(0.0, 1.0, size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# wald
# ---------------------------------------------------------------------------

class TestWald:
    def test_shape(self):
        result = cpr.wald(1.0, 1.0, size=(3, 3))
        assert result.shape == (3, 3)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.wald(1.0, 1.0, size=1000)
        assert result.get().min() > 0.0

    def test_dtype(self):
        result = cpr.wald(1.0, 1.0, size=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# zipf
# ---------------------------------------------------------------------------

class TestZipf:
    def test_shape(self):
        result = cpr.zipf(2.0, size=(3, 3))
        assert result.shape == (3, 3)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.zipf(2.0, size=1000)
        assert result.get().min() >= 1

    def test_dtype(self):
        result = cpr.zipf(2.0, size=5)
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# pareto
# ---------------------------------------------------------------------------

class TestPareto:
    def test_shape(self):
        result = cpr.pareto(2.0, size=(3, 3))
        assert result.shape == (3, 3)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.pareto(2.0, size=1000)
        assert result.get().min() >= 0.0

    def test_dtype(self):
        result = cpr.pareto(2.0, size=5)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# logseries
# ---------------------------------------------------------------------------

class TestLogseries:
    def test_shape(self):
        result = cpr.logseries(0.5, size=(3, 3))
        assert result.shape == (3, 3)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.logseries(0.5, size=1000)
        assert result.get().min() >= 1

    def test_dtype(self):
        result = cpr.logseries(0.5, size=5)
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# multinomial
# ---------------------------------------------------------------------------

class TestMultinomial:
    def test_shape(self):
        result = cpr.multinomial(10, [0.2, 0.3, 0.5])
        assert result.shape == (3,)

    def test_shape_with_size(self):
        result = cpr.multinomial(10, [0.2, 0.3, 0.5], size=4)
        assert result.shape == (4, 3)

    def test_sum(self):
        cpr.seed(0)
        result = cpr.multinomial(20, [0.25, 0.25, 0.25, 0.25])
        assert result.get().sum() == 20

    def test_dtype(self):
        result = cpr.multinomial(10, [0.5, 0.5])
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# multivariate_normal
# ---------------------------------------------------------------------------

class TestMultivariateNormal:
    def test_shape(self):
        mean = [0.0, 0.0]
        cov = [[1.0, 0.0], [0.0, 1.0]]
        result = cpr.multivariate_normal(mean, cov, size=5)
        assert result.shape == (5, 2)

    def test_dtype(self):
        mean = [0.0, 0.0]
        cov = [[1.0, 0.0], [0.0, 1.0]]
        result = cpr.multivariate_normal(mean, cov, size=3)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# dirichlet
# ---------------------------------------------------------------------------

class TestDirichlet:
    def test_shape(self):
        result = cpr.dirichlet([1.0, 2.0, 3.0], size=5)
        assert result.shape == (5, 3)

    def test_sums_to_one(self):
        cpr.seed(0)
        result = cpr.dirichlet([1.0, 1.0, 1.0], size=10)
        sums = result.get().sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_dtype(self):
        result = cpr.dirichlet([1.0, 1.0], size=3)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# f
# ---------------------------------------------------------------------------

class TestF:
    def test_shape(self):
        result = cpr.f(5, 10, size=(3, 3))
        assert result.shape == (3, 3)

    def test_positive(self):
        cpr.seed(0)
        result = cpr.f(5, 10, size=1000)
        assert result.get().min() >= 0.0

    def test_dtype(self):
        result = cpr.f(5, 10, size=5)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# hypergeometric
# ---------------------------------------------------------------------------

class TestHypergeometric:
    def test_shape(self):
        result = cpr.hypergeometric(10, 10, 10, size=(3, 3))
        assert result.shape == (3, 3)

    def test_range(self):
        cpr.seed(0)
        result = cpr.hypergeometric(10, 10, 10, size=1000)
        vals = result.get()
        assert vals.min() >= 0
        assert vals.max() <= 10

    def test_dtype(self):
        result = cpr.hypergeometric(10, 10, 10, size=5)
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# negative_binomial
# ---------------------------------------------------------------------------

class TestNegativeBinomial:
    def test_shape(self):
        result = cpr.negative_binomial(5, 0.5, size=(3, 3))
        assert result.shape == (3, 3)

    def test_non_negative(self):
        cpr.seed(0)
        result = cpr.negative_binomial(5, 0.5, size=1000)
        assert result.get().min() >= 0

    def test_dtype(self):
        result = cpr.negative_binomial(5, 0.5, size=5)
        assert result.dtype == np.int32
