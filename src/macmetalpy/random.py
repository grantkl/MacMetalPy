"""Random number generation (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from . import creation

_rng = np.random.RandomState()

def seed(s=None):
    """Seed the random number generator."""
    global _rng
    _rng = np.random.RandomState(s)

def rand(*shape):
    """Random values in a given shape from uniform [0, 1)."""
    return creation.array(np.asarray(_rng.rand(*shape), dtype=np.float32))

def randn(*shape):
    """Return sample(s) from standard normal distribution."""
    return creation.array(np.asarray(_rng.randn(*shape), dtype=np.float32))

def randint(low, high=None, size=None, dtype=int):
    """Return random integers from low (inclusive) to high (exclusive)."""
    result = _rng.randint(low, high, size=size)
    return creation.array(np.asarray(result, dtype=np.int32))

def random(size=None):
    """Return random floats in the half-open interval [0.0, 1.0)."""
    result = _rng.random_sample(size)
    return creation.array(np.asarray(result, dtype=np.float32))

def shuffle(a):
    """Modify array in-place by shuffling its contents."""
    from .ndarray import ndarray
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    tmp = a.get()
    _rng.shuffle(tmp)
    a.set(tmp)

def permutation(x):
    """Randomly permute a sequence or return a permuted range."""
    from .ndarray import ndarray
    if isinstance(x, ndarray):
        result = _rng.permutation(x.get())
    elif isinstance(x, int):
        result = _rng.permutation(x)
    else:
        result = _rng.permutation(np.asarray(x))
    return creation.array(np.asarray(result, dtype=np.int32 if isinstance(x, int) else np.float32))

def choice(a, size=None, replace=True, p=None):
    """Generate random sample from a given 1-D array."""
    from .ndarray import ndarray
    if isinstance(a, ndarray):
        a_np = a.get()
    elif isinstance(a, int):
        a_np = a
    else:
        a_np = np.asarray(a)
    p_np = p.get() if isinstance(p, ndarray) else p
    result = _rng.choice(a_np, size=size, replace=replace, p=p_np)
    return creation.array(np.asarray(result))

def normal(loc=0.0, scale=1.0, size=None):
    """Draw random samples from a normal distribution."""
    return creation.array(_rng.normal(loc, scale, size).astype(np.float32))

def uniform(low=0.0, high=1.0, size=None):
    """Draw samples from a uniform distribution."""
    return creation.array(_rng.uniform(low, high, size).astype(np.float32))

def beta(a, b, size=None):
    return creation.array(_rng.beta(a, b, size).astype(np.float32))

def binomial(n, p, size=None):
    return creation.array(np.asarray(_rng.binomial(n, p, size), dtype=np.int32))

def exponential(scale=1.0, size=None):
    return creation.array(_rng.exponential(scale, size).astype(np.float32))

def gamma(shape, scale=1.0, size=None):
    return creation.array(_rng.gamma(shape, scale, size).astype(np.float32))

def poisson(lam=1.0, size=None):
    return creation.array(np.asarray(_rng.poisson(lam, size), dtype=np.int32))

def standard_normal(size=None):
    return creation.array(_rng.standard_normal(size).astype(np.float32))

def standard_cauchy(size=None):
    return creation.array(_rng.standard_cauchy(size).astype(np.float32))

def standard_exponential(size=None):
    return creation.array(_rng.standard_exponential(size).astype(np.float32))

def standard_gamma(shape, size=None):
    return creation.array(_rng.standard_gamma(shape, size).astype(np.float32))

def standard_t(df, size=None):
    return creation.array(_rng.standard_t(df, size).astype(np.float32))

def chisquare(df, size=None):
    return creation.array(_rng.chisquare(df, size).astype(np.float32))

def geometric(p, size=None):
    return creation.array(np.asarray(_rng.geometric(p, size), dtype=np.int32))

def laplace(loc=0.0, scale=1.0, size=None):
    return creation.array(_rng.laplace(loc, scale, size).astype(np.float32))

def logistic(loc=0.0, scale=1.0, size=None):
    return creation.array(_rng.logistic(loc, scale, size).astype(np.float32))

def lognormal(mean=0.0, sigma=1.0, size=None):
    return creation.array(_rng.lognormal(mean, sigma, size).astype(np.float32))

def gumbel(loc=0.0, scale=1.0, size=None):
    return creation.array(_rng.gumbel(loc, scale, size).astype(np.float32))

def rayleigh(scale=1.0, size=None):
    return creation.array(_rng.rayleigh(scale, size).astype(np.float32))

def triangular(left, mode, right, size=None):
    return creation.array(_rng.triangular(left, mode, right, size).astype(np.float32))

def weibull(a, size=None):
    return creation.array(_rng.weibull(a, size).astype(np.float32))

def vonmises(mu, kappa, size=None):
    return creation.array(_rng.vonmises(mu, kappa, size).astype(np.float32))

def wald(mean, scale, size=None):
    return creation.array(_rng.wald(mean, scale, size).astype(np.float32))

def zipf(a, size=None):
    return creation.array(np.asarray(_rng.zipf(a, size), dtype=np.int32))

def pareto(a, size=None):
    return creation.array(_rng.pareto(a, size).astype(np.float32))

def logseries(p, size=None):
    return creation.array(np.asarray(_rng.logseries(p, size), dtype=np.int32))

def multinomial(n, pvals, size=None):
    result = _rng.multinomial(n, pvals, size)
    return creation.array(np.asarray(result, dtype=np.int32))

def multivariate_normal(mean, cov, size=None):
    from .ndarray import ndarray
    m = mean.get() if isinstance(mean, ndarray) else np.asarray(mean)
    c = cov.get() if isinstance(cov, ndarray) else np.asarray(cov)
    return creation.array(_rng.multivariate_normal(m, c, size).astype(np.float32))

def dirichlet(alpha, size=None):
    return creation.array(_rng.dirichlet(alpha, size).astype(np.float32))

def f(dfnum, dfden, size=None):
    return creation.array(_rng.f(dfnum, dfden, size).astype(np.float32))

def hypergeometric(ngood, nbad, nsample, size=None):
    return creation.array(np.asarray(_rng.hypergeometric(ngood, nbad, nsample, size), dtype=np.int32))

def negative_binomial(n, p, size=None):
    return creation.array(np.asarray(_rng.negative_binomial(n, p, size), dtype=np.int32))
