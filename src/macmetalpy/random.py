"""Random number generation (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from . import creation
from .ndarray import ndarray as _ndarray

# New Generator-based RNG for native float32 support
_rng_gen = np.random.default_rng()

# Keep legacy RandomState for get_state/set_state compatibility
_rng = np.random.RandomState()

def _fnd(x):
    """Fast ndarray construction from numpy array."""
    return _ndarray._from_np_direct(x)

def seed(s=None):
    """Seed the random number generator."""
    global _rng_gen, _rng
    _rng_gen = np.random.default_rng(s)
    _rng = np.random.RandomState(s)

def rand(*shape):
    """Random values in a given shape from uniform [0, 1)."""
    if not shape:
        shape = None
    return _fnd(_rng_gen.random(shape, dtype=np.float32))

def randn(*shape):
    """Return sample(s) from standard normal distribution."""
    if not shape:
        shape = None
    return _fnd(_rng_gen.standard_normal(shape, dtype=np.float32))

def randint(low, high=None, size=None, dtype=int):
    """Return random integers from low (inclusive) to high (exclusive)."""
    result = _rng_gen.integers(low, high, size=size, dtype=np.int32)
    return _fnd(np.asarray(result, dtype=np.int32))

def random(size=None):
    """Return random floats in the half-open interval [0.0, 1.0)."""
    return _fnd(_rng_gen.random(size, dtype=np.float32))

def shuffle(a):
    """Modify array in-place by shuffling its contents."""
    if not isinstance(a, _ndarray):
        a = creation.asarray(a)
    tmp = a.get()
    _rng_gen.shuffle(tmp)
    a.set(tmp)

def permutation(x):
    """Randomly permute a sequence or return a permuted range."""
    if isinstance(x, _ndarray):
        result = _rng_gen.permutation(x.get())
    elif isinstance(x, int):
        result = _rng_gen.permutation(x)
    else:
        result = _rng_gen.permutation(np.asarray(x))
    return _fnd(np.asarray(result, dtype=np.int32 if isinstance(x, int) else np.float32))

def choice(a, size=None, replace=True, p=None):
    """Generate random sample from a given 1-D array."""
    if isinstance(a, _ndarray):
        a_np = a.get()
    elif isinstance(a, int):
        a_np = a
    else:
        a_np = np.asarray(a)
    p_np = p.get() if isinstance(p, _ndarray) else p
    result = _rng_gen.choice(a_np, size=size, replace=replace, p=p_np)
    return _fnd(np.asarray(result))

def normal(loc=0.0, scale=1.0, size=None):
    """Draw random samples from a normal distribution."""
    return _fnd(_rng_gen.normal(loc, scale, size).astype(np.float32))

def uniform(low=0.0, high=1.0, size=None):
    """Draw samples from a uniform distribution."""
    return _fnd(_rng_gen.uniform(low, high, size).astype(np.float32))

def beta(a, b, size=None):
    return _fnd(_rng_gen.beta(a, b, size).astype(np.float32))

def binomial(n, p, size=None):
    return _fnd(np.asarray(_rng_gen.binomial(n, p, size), dtype=np.int32))

def exponential(scale=1.0, size=None):
    return _fnd(_rng_gen.exponential(scale, size).astype(np.float32))

def gamma(shape, scale=1.0, size=None):
    return _fnd(_rng_gen.gamma(shape, scale, size).astype(np.float32))

def poisson(lam=1.0, size=None):
    return _fnd(np.asarray(_rng_gen.poisson(lam, size), dtype=np.int32))

def standard_normal(size=None):
    return _fnd(_rng_gen.standard_normal(size, dtype=np.float32))

def standard_cauchy(size=None):
    return _fnd(_rng_gen.standard_cauchy(size).astype(np.float32))

def standard_exponential(size=None):
    return _fnd(_rng_gen.standard_exponential(size, dtype=np.float32))

def standard_gamma(shape, size=None):
    return _fnd(_rng_gen.standard_gamma(shape, size).astype(np.float32))

def standard_t(df, size=None):
    return _fnd(_rng_gen.standard_t(df, size).astype(np.float32))

def chisquare(df, size=None):
    return _fnd(_rng_gen.chisquare(df, size).astype(np.float32))

def geometric(p, size=None):
    return _fnd(np.asarray(_rng_gen.geometric(p, size), dtype=np.int32))

def laplace(loc=0.0, scale=1.0, size=None):
    return _fnd(_rng_gen.laplace(loc, scale, size).astype(np.float32))

def logistic(loc=0.0, scale=1.0, size=None):
    return _fnd(_rng_gen.logistic(loc, scale, size).astype(np.float32))

def lognormal(mean=0.0, sigma=1.0, size=None):
    return _fnd(_rng_gen.lognormal(mean, sigma, size).astype(np.float32))

def gumbel(loc=0.0, scale=1.0, size=None):
    return _fnd(_rng_gen.gumbel(loc, scale, size).astype(np.float32))

def rayleigh(scale=1.0, size=None):
    return _fnd(_rng_gen.rayleigh(scale, size).astype(np.float32))

def triangular(left, mode, right, size=None):
    return _fnd(_rng_gen.triangular(left, mode, right, size).astype(np.float32))

def weibull(a, size=None):
    return _fnd(_rng_gen.weibull(a, size).astype(np.float32))

def vonmises(mu, kappa, size=None):
    return _fnd(_rng_gen.vonmises(mu, kappa, size).astype(np.float32))

def wald(mean, scale, size=None):
    return _fnd(_rng_gen.wald(mean, scale, size).astype(np.float32))

def zipf(a, size=None):
    return _fnd(np.asarray(_rng_gen.zipf(a, size), dtype=np.int32))

def pareto(a, size=None):
    return _fnd(_rng_gen.pareto(a, size).astype(np.float32))

def logseries(p, size=None):
    return _fnd(np.asarray(_rng_gen.logseries(p, size), dtype=np.int32))

def multinomial(n, pvals, size=None):
    result = _rng_gen.multinomial(n, pvals, size)
    return _fnd(np.asarray(result, dtype=np.int32))

def multivariate_normal(mean, cov, size=None):
    m = mean.get() if isinstance(mean, _ndarray) else np.asarray(mean)
    c = cov.get() if isinstance(cov, _ndarray) else np.asarray(cov)
    return _fnd(_rng_gen.multivariate_normal(m, c, size).astype(np.float32))

def dirichlet(alpha, size=None):
    return _fnd(_rng_gen.dirichlet(alpha, size).astype(np.float32))

def f(dfnum, dfden, size=None):
    return _fnd(_rng_gen.f(dfnum, dfden, size).astype(np.float32))

def hypergeometric(ngood, nbad, nsample, size=None):
    return _fnd(np.asarray(_rng_gen.hypergeometric(ngood, nbad, nsample, size), dtype=np.int32))

def negative_binomial(n, p, size=None):
    return _fnd(np.asarray(_rng_gen.negative_binomial(n, p, size), dtype=np.int32))


# ── Aliases ────────────────────────────────────────────────────────────────

def random_sample(size=None):
    """Alias for random(). Return random floats in [0.0, 1.0)."""
    return random(size=size)

ranf = random
sample = random


# ── Additional distributions ──────────────────────────────────────────────

def power(a, size=None):
    """Draw samples from a power distribution. PDF: a * x^(a-1), 0 <= x <= 1."""
    result = _rng_gen.power(a, size).astype(np.float32)
    return _fnd(result)

def noncentral_chisquare(df, nonc, size=None):
    """Draw samples from a non-central chi-square distribution."""
    result = _rng_gen.noncentral_chisquare(df, nonc, size).astype(np.float32)
    return _fnd(result)

def noncentral_f(dfnum, dfden, nonc, size=None):
    """Draw samples from a non-central F distribution."""
    result = _rng_gen.noncentral_f(dfnum, dfden, nonc, size).astype(np.float32)
    return _fnd(result)


# ── permuted ──────────────────────────────────────────────────────────────

def permuted(x, axis=None, out=None):
    """Return a permuted copy of x without modifying the input."""
    if not isinstance(x, _ndarray):
        x = creation.asarray(x)
    tmp = x.get().copy()
    if axis is None:
        _rng_gen.shuffle(tmp.ravel())
        tmp = tmp.reshape(x.shape)
    else:
        np.apply_along_axis(lambda a: _rng_gen.shuffle(a) or a, axis, tmp)
    result = _fnd(tmp)
    if out is not None:
        out._np_data = result.get()
        out._buffer = None
        return out
    return result


# ── Generator / default_rng ──────────────────────────────────────────────

class Generator:
    """Simple Generator class wrapping module-level random functions."""

    def __init__(self, seed_val=None):
        if seed_val is not None:
            seed(seed_val)

    # Core
    def random(self, size=None):
        return random(size=size)

    def integers(self, low, high=None, size=None, dtype=int):
        return randint(low, high, size=size, dtype=dtype)

    # Distributions
    def normal(self, loc=0.0, scale=1.0, size=None):
        return normal(loc, scale, size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return uniform(low, high, size)

    def standard_normal(self, size=None):
        return standard_normal(size)

    def beta(self, a, b, size=None):
        return beta(a, b, size)

    def binomial(self, n, p, size=None):
        return binomial(n, p, size)

    def chisquare(self, df, size=None):
        return chisquare(df, size)

    def exponential(self, scale=1.0, size=None):
        return exponential(scale, size)

    def f(self, dfnum, dfden, size=None):
        return f(dfnum, dfden, size)

    def gamma(self, shape, scale=1.0, size=None):
        return gamma(shape, scale, size)

    def geometric(self, p, size=None):
        return geometric(p, size)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        return gumbel(loc, scale, size)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        return laplace(loc, scale, size)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        return logistic(loc, scale, size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        return lognormal(mean, sigma, size)

    def multinomial(self, n, pvals, size=None):
        return multinomial(n, pvals, size)

    def multivariate_normal(self, mean, cov, size=None):
        return multivariate_normal(mean, cov, size)

    def negative_binomial(self, n, p, size=None):
        return negative_binomial(n, p, size)

    def pareto(self, a, size=None):
        return pareto(a, size)

    def permutation(self, x):
        return permutation(x)

    def permuted(self, x, axis=None, out=None):
        return permuted(x, axis=axis, out=out)

    def poisson(self, lam=1.0, size=None):
        return poisson(lam, size)

    def power(self, a, size=None):
        return power(a, size)

    def rayleigh(self, scale=1.0, size=None):
        return rayleigh(scale, size)

    def shuffle(self, a):
        return shuffle(a)

    def standard_cauchy(self, size=None):
        return standard_cauchy(size)

    def standard_exponential(self, size=None):
        return standard_exponential(size)

    def standard_gamma(self, shape, size=None):
        return standard_gamma(shape, size)

    def standard_t(self, df, size=None):
        return standard_t(df, size)

    def triangular(self, left, mode, right, size=None):
        return triangular(left, mode, right, size)

    def vonmises(self, mu, kappa, size=None):
        return vonmises(mu, kappa, size)

    def wald(self, mean, scale, size=None):
        return wald(mean, scale, size)

    def weibull(self, a, size=None):
        return weibull(a, size)

    def zipf(self, a, size=None):
        return zipf(a, size)

    def dirichlet(self, alpha, size=None):
        return dirichlet(alpha, size)

    def choice(self, a, size=None, replace=True, p=None):
        return choice(a, size=size, replace=replace, p=p)

    def noncentral_chisquare(self, df, nonc, size=None):
        return noncentral_chisquare(df, nonc, size)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        return noncentral_f(dfnum, dfden, nonc, size)


def default_rng(seed_val=None):
    """Return a Generator object, optionally seeded."""
    return Generator(seed_val)


# ── State functions ────────────────────────────────────────────────────

def bytes(length):
    """Return random bytes."""
    return np.random.bytes(length)


def get_state():
    """Return the internal state of the global random number generator."""
    return _rng_gen.bit_generator.state


def set_state(state):
    """Set the internal state of the global random number generator."""
    _rng_gen.bit_generator.state = state


# ── BitGenerator / RNG classes (re-exports from numpy) ────────────────

from numpy.random import bit_generator, mtrand

try:
    from numpy.random import get_bit_generator, set_bit_generator
except ImportError:
    def get_bit_generator():
        raise NotImplementedError("get_bit_generator not available in this NumPy version")
    def set_bit_generator(bitgen):
        raise NotImplementedError("set_bit_generator not available in this NumPy version")


def random_integers(low, high=None, size=None):
    """Legacy random integers function (deprecated in NumPy)."""
    if high is None:
        high, low = low, 1
    return randint(low, high + 1, size=size)


BitGenerator = np.random.BitGenerator
MT19937 = np.random.MT19937
PCG64 = np.random.PCG64
PCG64DXSM = np.random.PCG64DXSM
Philox = np.random.Philox
RandomState = np.random.RandomState
SFC64 = np.random.SFC64
SeedSequence = np.random.SeedSequence
