"""Expanded benchmark functions for macmetalpy vs NumPy.

Covers all remaining computational APIs not in bench_core or bench_advanced.
"""

import math
import time

import numpy as np

import macmetalpy as cp

SIZE_MAP = {"small": 1_000, "medium": 100_000, "large": 1_000_000}
ALL_SIZES = ["small", "medium", "large"]

_WARMUP = 1
_REPEATS = 5


def _median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _time_mp_np(mp_fn, np_fn):
    """Run warmup + repeats for both mp and np functions, return median times."""
    for _ in range(_WARMUP):
        mp_fn()
    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        mp_fn()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    for _ in range(_WARMUP):
        np_fn()
    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        np_fn()
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


class _UnaryBench:
    """Picklable callable for benchmarking unary functions."""
    def __init__(self, cp_name, np_name, offset=0.1):
        self.cp_name = cp_name
        self.np_name = np_name
        self.offset = offset

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = np.random.rand(size).astype(np.float32) + self.offset
        a_cp = cp.array(a_np)
        return _time_mp_np(lambda: cp_f(a_cp).get(), lambda: np_f(a_np))


class _BinaryBench:
    """Picklable callable for benchmarking binary functions."""
    def __init__(self, cp_name, np_name, offset=0.1):
        self.cp_name = cp_name
        self.np_name = np_name
        self.offset = offset

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = np.random.rand(size).astype(np.float32) + self.offset
        b_np = np.random.rand(size).astype(np.float32) + self.offset
        a_cp, b_cp = cp.array(a_np), cp.array(b_np)
        return _time_mp_np(lambda: cp_f(a_cp, b_cp).get(), lambda: np_f(a_np, b_np))


class _ReductionBench:
    """Picklable callable for benchmarking reduction functions."""
    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = np.random.rand(size).astype(np.float32)
        a_cp = cp.array(a_np)
        def mp():
            r = cp_f(a_cp)
            if hasattr(r, 'get'): r.get()
        def np_run():
            np_f(a_np)
        return _time_mp_np(mp, np_run)


class _NanUnaryBench:
    """Picklable callable for benchmarking NaN-aware unary functions."""
    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = np.random.rand(size).astype(np.float32)
        a_np[np.random.rand(size) < 0.1] = np.nan
        a_cp = cp.array(a_np)
        def mp():
            r = cp_f(a_cp)
            if hasattr(r, 'get'): r.get()
        return _time_mp_np(mp, lambda: np_f(a_np))


class _IntBinaryBench:
    """Picklable callable for benchmarking integer binary functions."""
    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = np.random.randint(1, 1000, size=size).astype(np.int32)
        b_np = np.random.randint(1, 1000, size=size).astype(np.int32)
        a_cp, b_cp = cp.array(a_np), cp.array(b_np)
        return _time_mp_np(lambda: cp_f(a_cp, b_cp).get(), lambda: np_f(a_np, b_np))


class _IntUnaryBench:
    """Picklable callable for benchmarking integer unary functions."""
    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = np.random.randint(1, 1000, size=size).astype(np.int32)
        a_cp = cp.array(a_np)
        return _time_mp_np(lambda: cp_f(a_cp).get(), lambda: np_f(a_np))


def _resolve(module, name):
    """Resolve a dotted name like 'linalg.norm' on a module."""
    obj = module
    for part in name.split('.'):
        obj = getattr(obj, part)
    return obj


# ===================================================================
# Trig
# ===================================================================
bench_arcsin = _UnaryBench('arcsin', 'arcsin', offset=0.0)  # input in [-1,1]
bench_arccos = _UnaryBench('arccos', 'arccos', offset=0.0)
bench_arctan = _UnaryBench('arctan', 'arctan')
bench_arctan2 = _BinaryBench('arctan2', 'arctan2')
bench_sinh = _UnaryBench('sinh', 'sinh')
bench_cosh = _UnaryBench('cosh', 'cosh')
bench_tanh = _UnaryBench('tanh', 'tanh')
bench_arcsinh = _UnaryBench('arcsinh', 'arcsinh')
bench_arctanh = _UnaryBench('arctanh', 'arctanh', offset=0.0)  # input in (-1,1)
bench_degrees = _UnaryBench('degrees', 'degrees')
bench_radians = _UnaryBench('radians', 'radians')
bench_deg2rad = _UnaryBench('deg2rad', 'deg2rad')
bench_rad2deg = _UnaryBench('rad2deg', 'rad2deg')
bench_hypot = _BinaryBench('hypot', 'hypot')


def bench_arccosh(size):
    a_np = np.random.rand(size).astype(np.float32) + 1.0  # arccosh needs x >= 1
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.arccosh(a_cp).get(), lambda: np.arccosh(a_np))


# ===================================================================
# Ufuncs extra
# ===================================================================
bench_exp2 = _UnaryBench('exp2', 'exp2')
bench_expm1 = _UnaryBench('expm1', 'expm1')
bench_log1p = _UnaryBench('log1p', 'log1p')
bench_log2 = _UnaryBench('log2', 'log2')
bench_log10 = _UnaryBench('log10', 'log10')
bench_cbrt = _UnaryBench('cbrt', 'cbrt')
bench_reciprocal = _UnaryBench('reciprocal', 'reciprocal')
bench_rint = _UnaryBench('rint', 'rint')
bench_trunc = _UnaryBench('trunc', 'trunc')
bench_sign = _UnaryBench('sign', 'sign')
bench_floor = _UnaryBench('floor', 'floor')
bench_ceil = _UnaryBench('ceil', 'ceil')
bench_square = _UnaryBench('square', 'square')
bench_negative = _UnaryBench('negative', 'negative')
bench_positive = _UnaryBench('positive', 'positive')
bench_absolute = _UnaryBench('absolute', 'absolute')
bench_fabs = _UnaryBench('fabs', 'fabs')
bench_float_power = _BinaryBench('float_power', 'float_power')
bench_copysign = _BinaryBench('copysign', 'copysign')
bench_nextafter = _BinaryBench('nextafter', 'nextafter')
bench_signbit = _UnaryBench('signbit', 'signbit')
bench_fmax = _BinaryBench('fmax', 'fmax')
bench_fmin = _BinaryBench('fmin', 'fmin')
bench_fmod = _BinaryBench('fmod', 'fmod')
bench_remainder = _BinaryBench('remainder', 'remainder')
bench_true_divide = _BinaryBench('true_divide', 'true_divide')
bench_logaddexp = _BinaryBench('logaddexp', 'logaddexp')
bench_logaddexp2 = _BinaryBench('logaddexp2', 'logaddexp2')
bench_maximum = _BinaryBench('maximum', 'maximum')
bench_minimum = _BinaryBench('minimum', 'minimum')


def bench_heaviside(size):
    a_np = np.random.rand(size).astype(np.float32) - 0.5
    h_np = np.full(size, 0.5, dtype=np.float32)
    a_cp, h_cp = cp.array(a_np), cp.array(h_np)
    return _time_mp_np(lambda: cp.heaviside(a_cp, h_cp).get(), lambda: np.heaviside(a_np, h_np))


def bench_modf(size):
    a_np = (np.random.rand(size).astype(np.float32) * 10)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.modf(a_cp)
        r[0].get()
    return _time_mp_np(mp, lambda: np.modf(a_np))


def bench_ldexp(size):
    a_np = np.random.rand(size).astype(np.float32)
    b_np = np.random.randint(0, 10, size=size).astype(np.int32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.ldexp(a_cp, b_cp).get(), lambda: np.ldexp(a_np, b_np))


def bench_frexp(size):
    a_np = np.random.rand(size).astype(np.float32) + 0.1
    a_cp = cp.array(a_np)
    def mp():
        r = cp.frexp(a_cp)
        r[0].get()
    return _time_mp_np(mp, lambda: np.frexp(a_np))


# ===================================================================
# Bitwise
# ===================================================================
bench_bitwise_and = _IntBinaryBench('bitwise_and', 'bitwise_and')
bench_bitwise_or = _IntBinaryBench('bitwise_or', 'bitwise_or')
bench_bitwise_xor = _IntBinaryBench('bitwise_xor', 'bitwise_xor')
bench_invert = _IntUnaryBench('invert', 'invert')
bench_left_shift = _IntBinaryBench('left_shift', 'left_shift')
bench_right_shift = _IntBinaryBench('right_shift', 'right_shift')
bench_gcd = _IntBinaryBench('gcd', 'gcd')
bench_lcm = _IntBinaryBench('lcm', 'lcm')


def bench_bitwise_count(size):
    a_np = np.random.randint(0, 2**31, size=size).astype(np.int32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.bitwise_count(a_cp).get(), lambda: np.bitwise_count(a_np))


# ===================================================================
# Logic
# ===================================================================
class _BoolBinaryBench:
    """Picklable callable for benchmarking boolean binary functions."""
    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = (np.random.rand(size) > 0.5)
        b_np = (np.random.rand(size) > 0.5)
        a_cp, b_cp = cp.array(a_np), cp.array(b_np)
        return _time_mp_np(lambda: cp_f(a_cp, b_cp).get(), lambda: np_f(a_np, b_np))


class _BoolUnaryBench:
    """Picklable callable for benchmarking boolean unary functions."""
    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = (np.random.rand(size) > 0.5)
        a_cp = cp.array(a_np)
        return _time_mp_np(lambda: cp_f(a_cp).get(), lambda: np_f(a_np))


bench_logical_and = _BoolBinaryBench('logical_and', 'logical_and')
bench_logical_or = _BoolBinaryBench('logical_or', 'logical_or')
bench_logical_not = _BoolUnaryBench('logical_not', 'logical_not')
bench_logical_xor = _BoolBinaryBench('logical_xor', 'logical_xor')
bench_isnan = _UnaryBench('isnan', 'isnan')
bench_isinf = _UnaryBench('isinf', 'isinf')
bench_isfinite = _UnaryBench('isfinite', 'isfinite')


def bench_isclose(size):
    a_np = np.random.rand(size).astype(np.float32)
    b_np = a_np + np.random.rand(size).astype(np.float32) * 1e-5
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.isclose(a_cp, b_cp).get(), lambda: np.isclose(a_np, b_np))


def bench_allclose(size):
    a_np = np.random.rand(size).astype(np.float32)
    b_np = a_np + np.random.rand(size).astype(np.float32) * 1e-8
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.allclose(a_cp, b_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.allclose(a_np, b_np))


def bench_array_equal(size):
    a_np = np.random.rand(size).astype(np.float32)
    b_np = a_np.copy()
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.array_equal(a_cp, b_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.array_equal(a_np, b_np))


bench_isneginf = _UnaryBench('isneginf', 'isneginf')
bench_isposinf = _UnaryBench('isposinf', 'isposinf')
bench_greater_equal = _BinaryBench('greater_equal', 'greater_equal')
bench_less_equal = _BinaryBench('less_equal', 'less_equal')


def bench_iscomplex(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.iscomplex(a_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.iscomplex(a_np))


def bench_isreal(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.isreal(a_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.isreal(a_np))


# ===================================================================
# Window functions
# ===================================================================
class _WindowBench:
    """Picklable callable for benchmarking window functions."""
    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        return _time_mp_np(
            lambda: cp_f(size).get(),
            lambda: np_f(size),
        )


bench_bartlett = _WindowBench('bartlett', 'bartlett')
bench_blackman = _WindowBench('blackman', 'blackman')
bench_hamming = _WindowBench('hamming', 'hamming')
bench_hanning = _WindowBench('hanning', 'hanning')


def bench_kaiser(size):
    return _time_mp_np(
        lambda: cp.kaiser(size, 14.0).get(),
        lambda: np.kaiser(size, 14.0),
    )


# ===================================================================
# Complex ops
# ===================================================================
def bench_angle(size):
    a_np = (np.random.rand(size) + 1j * np.random.rand(size)).astype(np.complex64)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.angle(a_cp).get(), lambda: np.angle(a_np))


def bench_real(size):
    a_np = (np.random.rand(size) + 1j * np.random.rand(size)).astype(np.complex64)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.real(a_cp).get(), lambda: np.real(a_np))


def bench_imag(size):
    a_np = (np.random.rand(size) + 1j * np.random.rand(size)).astype(np.complex64)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.imag(a_cp).get(), lambda: np.imag(a_np))


def bench_conj(size):
    a_np = (np.random.rand(size) + 1j * np.random.rand(size)).astype(np.complex64)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.conj(a_cp).get(), lambda: np.conj(a_np))


def bench_real_if_close(size):
    a_np = np.random.rand(size).astype(np.float32) + 0j
    a_cp = cp.array(a_np)
    def mp():
        r = cp.real_if_close(a_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.real_if_close(a_np))


# ===================================================================
# Manipulation extra
# ===================================================================
def bench_pad(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.pad(a_cp, 10, mode='constant').get(),
        lambda: np.pad(a_np, 10, mode='constant'),
    )


def bench_delete(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    idx = np.array([0, size // 4, size // 2])
    return _time_mp_np(
        lambda: cp.delete(a_cp, idx).get(),
        lambda: np.delete(a_np, idx),
    )


def bench_insert(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.insert(a_cp, size // 2, 0.0).get(),
        lambda: np.insert(a_np, size // 2, 0.0),
    )


def bench_block(size):
    half = size // 2
    a_np = np.random.rand(half).astype(np.float32)
    b_np = np.random.rand(half).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(
        lambda: cp.block([a_cp, b_cp]).get(),
        lambda: np.block([a_np, b_np]),
    )


def bench_broadcast_to(size):
    a_np = np.random.rand(1).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.broadcast_to(a_cp, (size,)).get(),
        lambda: np.broadcast_to(a_np, (size,)),
    )


def bench_moveaxis(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.moveaxis(a_cp, 0, 1).get(),
        lambda: np.moveaxis(a_np, 0, 1),
    )


def bench_swapaxes(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.swapaxes(a_cp, 0, 1).get(),
        lambda: np.swapaxes(a_np, 0, 1),
    )


def bench_rollaxis(size):
    n = int(size ** (1/3))
    a_np = np.random.rand(n, n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.rollaxis(a_cp, 2, 0).get(),
        lambda: np.rollaxis(a_np, 2, 0),
    )


def bench_atleast_1d(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.atleast_1d(a_cp).get(),
        lambda: np.atleast_1d(a_np),
    )


def bench_atleast_2d(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.atleast_2d(a_cp).get(),
        lambda: np.atleast_2d(a_np),
    )


def bench_atleast_3d(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.atleast_3d(a_cp).get(),
        lambda: np.atleast_3d(a_np),
    )


def bench_dstack(size):
    half = size // 2
    a_np = np.random.rand(half).astype(np.float32)
    b_np = np.random.rand(half).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(
        lambda: cp.dstack([a_cp, b_cp]).get(),
        lambda: np.dstack([a_np, b_np]),
    )


def bench_column_stack(size):
    half = size // 2
    a_np = np.random.rand(half).astype(np.float32)
    b_np = np.random.rand(half).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(
        lambda: cp.column_stack([a_cp, b_cp]).get(),
        lambda: np.column_stack([a_np, b_np]),
    )


def bench_fliplr(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.fliplr(a_cp).get(), lambda: np.fliplr(a_np))


def bench_flipud(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.flipud(a_cp).get(), lambda: np.flipud(a_np))


def bench_rot90(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.rot90(a_cp).get(), lambda: np.rot90(a_np))


def bench_copyto(size):
    a_np = np.random.rand(size).astype(np.float32)
    b_np = np.empty_like(a_np)
    a_cp = cp.array(a_np)
    b_cp = cp.empty(size, dtype=np.float32)
    def mp():
        cp.copyto(b_cp, a_cp)
        b_cp.get()
    def np_run():
        np.copyto(b_np, a_np)
    return _time_mp_np(mp, np_run)


def bench_trim_zeros(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_np[:10] = 0; a_np[-10:] = 0
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.trim_zeros(a_cp).get(),
        lambda: np.trim_zeros(a_np),
    )


def bench_resize(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.resize(a_cp, (size * 2,)).get(),
        lambda: np.resize(a_np, (size * 2,)),
    )


def bench_append(size):
    a_np = np.random.rand(size).astype(np.float32)
    b_np = np.random.rand(100).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(
        lambda: cp.append(a_cp, b_cp).get(),
        lambda: np.append(a_np, b_np),
    )


def bench_broadcast_arrays(size):
    a_np = np.random.rand(size, 1).astype(np.float32)
    b_np = np.random.rand(1, 10).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.broadcast_arrays(a_cp, b_cp)
        r[0].get()
    def np_run():
        np.broadcast_arrays(a_np, b_np)
    return _time_mp_np(mp, np_run)


def bench_asfortranarray(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.asfortranarray(a_cp).get(),
        lambda: np.asfortranarray(a_np),
    )


def bench_ascontiguousarray(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.ascontiguousarray(a_cp).get(),
        lambda: np.ascontiguousarray(a_np),
    )


def bench_dsplit(size):
    n = max(int(size ** (1/3)), 4)
    a_np = np.random.rand(n, n, 4).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.dsplit(a_cp, 2)
        r[0].get()
    def np_run():
        np.dsplit(a_np, 2)
    return _time_mp_np(mp, np_run)


def bench_hsplit(size):
    n = int(size ** 0.5)
    n = n - n % 2
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.hsplit(a_cp, 2)
        r[0].get()
    def np_run():
        np.hsplit(a_np, 2)
    return _time_mp_np(mp, np_run)


def bench_vsplit(size):
    n = int(size ** 0.5)
    n = n - n % 2
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.vsplit(a_cp, 2)
        r[0].get()
    def np_run():
        np.vsplit(a_np, 2)
    return _time_mp_np(mp, np_run)


def bench_copy(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.copy(a_cp).get(), lambda: np.copy(a_np))


# ===================================================================
# Creation extra
# ===================================================================
def bench_fromfunction(size):
    n = int(size ** 0.5)
    def mp():
        r = cp.fromfunction(lambda i, j: i + j, (n, n), dtype=np.float32)
        r.get()
    def np_run():
        np.fromfunction(lambda i, j: i + j, (n, n), dtype=np.float32)
    return _time_mp_np(mp, np_run)


def bench_diagflat(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.diagflat(a_cp).get(), lambda: np.diagflat(a_np))


def bench_vander(size):
    n = min(size, 1000)
    a_np = np.random.rand(n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.vander(a_cp, 5).get(), lambda: np.vander(a_np, 5))


def bench_geomspace(size):
    return _time_mp_np(
        lambda: cp.geomspace(1, 1000, size, dtype=np.float32).get(),
        lambda: np.geomspace(1, 1000, size, dtype=np.float32),
    )


def bench_asarray(size):
    data = np.random.rand(size).astype(np.float32)
    return _time_mp_np(lambda: cp.asarray(data).get(), lambda: np.asarray(data))


def bench_asanyarray(size):
    data = np.random.rand(size).astype(np.float32)
    return _time_mp_np(lambda: cp.asanyarray(data).get(), lambda: np.asanyarray(data))


def bench_asarray_chkfinite(size):
    data = np.random.rand(size).astype(np.float32)
    return _time_mp_np(lambda: cp.asarray_chkfinite(data).get(), lambda: np.asarray_chkfinite(data))


def bench_identity(size):
    n = int(size ** 0.5)
    return _time_mp_np(
        lambda: cp.identity(n, dtype=np.float32).get(),
        lambda: np.identity(n, dtype=np.float32),
    )


def bench_tri(size):
    n = int(size ** 0.5)
    return _time_mp_np(
        lambda: cp.tri(n, dtype=np.float32).get(),
        lambda: np.tri(n, dtype=np.float32),
    )


def bench_triu(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.triu(a_cp).get(), lambda: np.triu(a_np))


def bench_tril(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.tril(a_cp).get(), lambda: np.tril(a_np))


def bench_diag(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.diag(a_cp).get(), lambda: np.diag(a_np))


def bench_indices(size):
    n = int(size ** 0.5)
    n = min(n, 500)
    def mp():
        r = cp.indices((n, n), dtype=np.float32)
        r[0].get()
    def np_run():
        np.indices((n, n), dtype=np.float32)
    return _time_mp_np(mp, np_run)


def bench_meshgrid(size):
    n = int(size ** 0.5)
    a_np = np.arange(n, dtype=np.float32)
    b_np = np.arange(n, dtype=np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.meshgrid(a_cp, b_cp)
        r[0].get()
    def np_run():
        np.meshgrid(a_np, b_np)
    return _time_mp_np(mp, np_run)


def bench_logspace(size):
    return _time_mp_np(
        lambda: cp.logspace(0, 2, size, dtype=np.float32).get(),
        lambda: np.logspace(0, 2, size, dtype=np.float32),
    )


def bench_around(size):
    a_np = np.random.rand(size).astype(np.float32) * 100
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.around(a_cp, 2).get(), lambda: np.around(a_np, 2))


# ===================================================================
# Reduction extra
# ===================================================================
bench_count_nonzero = _ReductionBench('count_nonzero', 'count_nonzero')
bench_amax = _ReductionBench('amax', 'amax')
bench_amin = _ReductionBench('amin', 'amin')


def bench_ptp(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.ptp(a_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.ptp(a_np))


def bench_average(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.average(a_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.average(a_np))


def bench_percentile(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.percentile(a_cp, 50)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.percentile(a_np, 50))


def bench_quantile(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.quantile(a_cp, 0.5)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.quantile(a_np, 0.5))


def bench_median(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.median(a_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.median(a_np))


def bench_divmod_fn(size):
    a_np = np.random.rand(size).astype(np.float32) + 0.1
    b_np = np.random.rand(size).astype(np.float32) + 0.1
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.divmod(a_cp, b_cp)
        r[0].get()
    def np_run():
        np.divmod(a_np, b_np)
    return _time_mp_np(mp, np_run)


# ===================================================================
# NaN extra
# ===================================================================
bench_nanprod = _NanUnaryBench('nanprod', 'nanprod')
bench_nancumsum = _NanUnaryBench('nancumsum', 'nancumsum')
bench_nancumprod = _NanUnaryBench('nancumprod', 'nancumprod')
bench_nanargmax = _NanUnaryBench('nanargmax', 'nanargmax')
bench_nanargmin = _NanUnaryBench('nanargmin', 'nanargmin')
bench_nanvar = _NanUnaryBench('nanvar', 'nanvar')
bench_nanmedian = _NanUnaryBench('nanmedian', 'nanmedian')


def bench_nanpercentile(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_np[np.random.rand(size) < 0.1] = np.nan
    a_cp = cp.array(a_np)
    def mp():
        r = cp.nanpercentile(a_cp, 50)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.nanpercentile(a_np, 50))


def bench_nanquantile(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_np[np.random.rand(size) < 0.1] = np.nan
    a_cp = cp.array(a_np)
    def mp():
        r = cp.nanquantile(a_cp, 0.5)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.nanquantile(a_np, 0.5))


# ===================================================================
# Indexing extra
# ===================================================================
def bench_fill_diagonal(size):
    n = int(size ** 0.5)
    a_np = np.zeros((n, n), dtype=np.float32)
    a_cp = cp.zeros((n, n), dtype=np.float32)
    def mp():
        cp.fill_diagonal(a_cp, 1.0)
        a_cp.get()
    def np_run():
        np.fill_diagonal(a_np, 1.0)
    return _time_mp_np(mp, np_run)


def bench_diag_indices(size):
    n = int(size ** 0.5)
    def mp():
        r = cp.diag_indices(n)
        r[0].get()
    def np_run():
        np.diag_indices(n)
    return _time_mp_np(mp, np_run)


def bench_diag_indices_from(size):
    n = int(size ** 0.5)
    a_np = np.zeros((n, n), dtype=np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.diag_indices_from(a_cp)
        r[0].get()
    def np_run():
        np.diag_indices_from(a_np)
    return _time_mp_np(mp, np_run)


def bench_tril_indices(size):
    n = int(size ** 0.5)
    n = min(n, 500)
    def mp():
        r = cp.tril_indices(n)
        r[0].get()
    def np_run():
        np.tril_indices(n)
    return _time_mp_np(mp, np_run)


def bench_triu_indices(size):
    n = int(size ** 0.5)
    n = min(n, 500)
    def mp():
        r = cp.triu_indices(n)
        r[0].get()
    def np_run():
        np.triu_indices(n)
    return _time_mp_np(mp, np_run)


def bench_ravel_multi_index(size):
    n = int(size ** 0.5)
    idx = (np.random.randint(0, n, size=size).astype(np.intp),
           np.random.randint(0, n, size=size).astype(np.intp))
    idx_cp = (cp.array(idx[0]), cp.array(idx[1]))
    def mp():
        r = cp.ravel_multi_index(idx_cp, (n, n))
        r.get()
    def np_run():
        np.ravel_multi_index(idx, (n, n))
    return _time_mp_np(mp, np_run)


def bench_unravel_index(size):
    n = int(size ** 0.5)
    idx_np = np.random.randint(0, n * n, size=size).astype(np.intp)
    idx_cp = cp.array(idx_np)
    def mp():
        r = cp.unravel_index(idx_cp, (n, n))
        r[0].get()
    def np_run():
        np.unravel_index(idx_np, (n, n))
    return _time_mp_np(mp, np_run)


def bench_flatnonzero(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_np[a_np < 0.5] = 0.0
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.flatnonzero(a_cp).get(), lambda: np.flatnonzero(a_np))


def bench_choose(size):
    n = min(size, 10000)
    idx_np = np.random.randint(0, 3, size=n).astype(np.intp)
    choices_np = [np.random.rand(n).astype(np.float32) for _ in range(3)]
    idx_cp = cp.array(idx_np)
    choices_cp = [cp.array(c) for c in choices_np]
    return _time_mp_np(
        lambda: cp.choose(idx_cp, choices_cp).get(),
        lambda: np.choose(idx_np, choices_np),
    )


def bench_compress(size):
    a_np = np.random.rand(size).astype(np.float32)
    cond_np = np.random.rand(size) > 0.5
    a_cp = cp.array(a_np)
    cond_cp = cp.array(cond_np)
    return _time_mp_np(
        lambda: cp.compress(cond_cp, a_cp).get(),
        lambda: np.compress(cond_np, a_np),
    )


def bench_select(size):
    a_np = np.random.rand(size).astype(np.float32)
    cond1 = a_np > 0.7
    cond2 = a_np < 0.3
    a_cp = cp.array(a_np)
    cond1_cp, cond2_cp = cp.array(cond1), cp.array(cond2)
    return _time_mp_np(
        lambda: cp.select([cond1_cp, cond2_cp], [a_cp, a_cp * 2], default=0.0).get(),
        lambda: np.select([cond1, cond2], [a_np, a_np * 2], default=0.0),
    )


def bench_extract(size):
    a_np = np.random.rand(size).astype(np.float32)
    cond_np = a_np > 0.5
    a_cp = cp.array(a_np)
    cond_cp = cp.array(cond_np)
    return _time_mp_np(
        lambda: cp.extract(cond_cp, a_cp).get(),
        lambda: np.extract(cond_np, a_np),
    )


def bench_putmask(size):
    a_np = np.random.rand(size).astype(np.float32)
    mask_np = np.random.rand(size) > 0.5
    vals_np = np.float32(0.0)
    a_cp = cp.array(a_np.copy())
    mask_cp = cp.array(mask_np)
    def mp():
        cp.putmask(a_cp, mask_cp, vals_np)
        a_cp.get()
    def np_run():
        a_tmp = a_np.copy()
        np.putmask(a_tmp, mask_np, vals_np)
    return _time_mp_np(mp, np_run)


def bench_place(size):
    a_np = np.random.rand(size).astype(np.float32)
    mask_np = np.random.rand(size) > 0.5
    a_cp = cp.array(a_np.copy())
    mask_cp = cp.array(mask_np)
    vals = np.array([0.0], dtype=np.float32)
    vals_cp = cp.array(vals)
    def mp():
        cp.place(a_cp, mask_cp, vals_cp)
        a_cp.get()
    def np_run():
        a_tmp = a_np.copy()
        np.place(a_tmp, mask_np, vals)
    return _time_mp_np(mp, np_run)


def bench_take_along_axis(size):
    a_np = np.random.rand(size).astype(np.float32)
    idx_np = np.argsort(a_np)[:size // 10].astype(np.intp)
    a_cp = cp.array(a_np)
    idx_cp = cp.array(idx_np)
    return _time_mp_np(
        lambda: cp.take_along_axis(a_cp, idx_cp, axis=0).get(),
        lambda: np.take_along_axis(a_np, idx_np, axis=0),
    )


def bench_put_along_axis(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    idx_np = np.zeros((n, 1), dtype=np.intp)
    vals_np = np.ones((n, 1), dtype=np.float32)
    a_cp = cp.array(a_np.copy())
    idx_cp = cp.array(idx_np)
    vals_cp = cp.array(vals_np)
    def mp():
        cp.put_along_axis(a_cp, idx_cp, vals_cp, axis=1)
        a_cp.get()
    def np_run():
        a_tmp = a_np.copy()
        np.put_along_axis(a_tmp, idx_np, vals_np, axis=1)
    return _time_mp_np(mp, np_run)


def bench_ix_(size):
    n = int(size ** 0.5)
    a_np = np.arange(n, dtype=np.intp)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.ix_(a_cp, a_cp)
        r[0].get()
    def np_run():
        np.ix_(a_np, a_np)
    return _time_mp_np(mp, np_run)


def bench_mask_indices(size):
    n = int(size ** 0.5)
    n = min(n, 500)
    def mp():
        r = cp.mask_indices(n, cp.triu)
        r[0].get()
    def np_run():
        np.mask_indices(n, np.triu)
    return _time_mp_np(mp, np_run)


# ===================================================================
# Set extra
# ===================================================================
def bench_setxor1d(size):
    a_np = np.random.randint(0, size, size=size).astype(np.int32)
    b_np = np.random.randint(0, size, size=size).astype(np.int32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(
        lambda: cp.setxor1d(a_cp, b_cp).get(),
        lambda: np.setxor1d(a_np, b_np),
    )


# ===================================================================
# Math ext
# ===================================================================
def bench_sinc(size):
    a_np = np.random.rand(size).astype(np.float32) * 10
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.sinc(a_cp).get(), lambda: np.sinc(a_np))


def bench_convolve(size):
    n = min(size, 10000)
    a_np = np.random.rand(n).astype(np.float32)
    b_np = np.random.rand(min(100, n)).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(
        lambda: cp.convolve(a_cp, b_cp).get(),
        lambda: np.convolve(a_np, b_np),
    )


def bench_interp(size):
    x_np = np.sort(np.random.rand(size).astype(np.float64))
    xp_np = np.linspace(0, 1, 100, dtype=np.float64)
    fp_np = np.random.rand(100).astype(np.float64)
    x_cp, xp_cp, fp_cp = cp.array(x_np), cp.array(xp_np), cp.array(fp_np)
    return _time_mp_np(
        lambda: cp.interp(x_cp, xp_cp, fp_cp).get(),
        lambda: np.interp(x_np, xp_np, fp_np),
    )


def bench_gradient(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.gradient(a_cp)
        if isinstance(r, list): r[0].get()
        else: r.get()
    def np_run():
        np.gradient(a_np)
    return _time_mp_np(mp, np_run)


def bench_trapezoid(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.trapezoid(a_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.trapezoid(a_np))


def bench_i0(size):
    a_np = np.random.rand(size).astype(np.float64) * 5
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.i0(a_cp).get(), lambda: np.i0(a_np))


def bench_fix(size):
    a_np = (np.random.rand(size).astype(np.float32) - 0.5) * 100
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.fix(a_cp).get(), lambda: np.fix(a_np))


def bench_unwrap(size):
    a_np = np.random.rand(size).astype(np.float64) * 4 * np.pi
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.unwrap(a_cp).get(), lambda: np.unwrap(a_np))


def bench_spacing(size):
    a_np = np.random.rand(size).astype(np.float64) + 0.1
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.spacing(a_cp).get(), lambda: np.spacing(a_np))


def bench_diff(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.diff(a_cp).get(), lambda: np.diff(a_np))


def bench_ediff1d(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.ediff1d(a_cp).get(), lambda: np.ediff1d(a_np))


# ===================================================================
# Stats
# ===================================================================
def bench_histogram(size):
    a_np = np.random.rand(size).astype(np.float64)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.histogram(a_cp, bins=50)
        r[0].get()
    def np_run():
        np.histogram(a_np, bins=50)
    return _time_mp_np(mp, np_run)


def bench_histogram2d(size):
    n = min(size, 100000)
    a_np = np.random.rand(n).astype(np.float64)
    b_np = np.random.rand(n).astype(np.float64)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.histogram2d(a_cp, b_cp, bins=50)
        r[0].get()
    def np_run():
        np.histogram2d(a_np, b_np, bins=50)
    return _time_mp_np(mp, np_run)


def bench_histogramdd(size):
    n = min(size, 50000)
    a_np = np.random.rand(n, 3).astype(np.float64)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.histogramdd(a_cp, bins=10)
        r[0].get()
    def np_run():
        np.histogramdd(a_np, bins=10)
    return _time_mp_np(mp, np_run)


def bench_histogram_bin_edges(size):
    a_np = np.random.rand(size).astype(np.float64)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.histogram_bin_edges(a_cp, bins=50)
        r.get()
    def np_run():
        np.histogram_bin_edges(a_np, bins=50)
    return _time_mp_np(mp, np_run)


def bench_bincount(size):
    a_np = np.random.randint(0, 1000, size=size).astype(np.intp)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.bincount(a_cp).get(), lambda: np.bincount(a_np))


def bench_digitize(size):
    a_np = np.random.rand(size).astype(np.float64)
    bins_np = np.linspace(0, 1, 50, dtype=np.float64)
    a_cp = cp.array(a_np)
    bins_cp = cp.array(bins_np)
    return _time_mp_np(
        lambda: cp.digitize(a_cp, bins_cp).get(),
        lambda: np.digitize(a_np, bins_np),
    )


def bench_corrcoef(size):
    n = min(size, 10000)
    a_np = np.random.rand(n).astype(np.float64)
    b_np = np.random.rand(n).astype(np.float64)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.corrcoef(a_cp, b_cp)
        r.get()
    def np_run():
        np.corrcoef(a_np, b_np)
    return _time_mp_np(mp, np_run)


def bench_correlate(size):
    n = min(size, 10000)
    a_np = np.random.rand(n).astype(np.float32)
    b_np = np.random.rand(min(100, n)).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(
        lambda: cp.correlate(a_cp, b_cp).get(),
        lambda: np.correlate(a_np, b_np),
    )


def bench_cov(size):
    n = min(size, 10000)
    a_np = np.random.rand(n).astype(np.float64)
    b_np = np.random.rand(n).astype(np.float64)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.cov(a_cp, b_cp)
        r.get()
    def np_run():
        np.cov(a_np, b_np)
    return _time_mp_np(mp, np_run)


# ===================================================================
# Linalg extra
# ===================================================================
def bench_lstsq(size):
    n = min(int(size ** 0.5), 500)
    a_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.linalg.lstsq(a_cp, b_cp, rcond=None)
        r[0].get()
    def np_run():
        np.linalg.lstsq(a_np, b_np, rcond=None)
    return _time_mp_np(mp, np_run)


def bench_pinv(size):
    n = min(int(size ** 0.5), 500)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.linalg.pinv(a_cp).get(), lambda: np.linalg.pinv(a_np))


def bench_matrix_rank(size):
    n = min(int(size ** 0.5), 500)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.linalg.matrix_rank(a_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.linalg.matrix_rank(a_np))


def bench_matrix_power(size):
    n = min(int(size ** 0.5), 200)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.linalg.matrix_power(a_cp, 3).get(), lambda: np.linalg.matrix_power(a_np, 3))


def bench_eigvals(size):
    n = min(int(size ** 0.5), 500)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.linalg.eigvals(a_cp).get(), lambda: np.linalg.eigvals(a_np))


def bench_cond(size):
    n = min(int(size ** 0.5), 500)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_np = a_np @ a_np.T + n * np.eye(n, dtype=np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.linalg.cond(a_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.linalg.cond(a_np))


def bench_multi_dot(size):
    n = min(int(size ** 0.5), 300)
    a_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n, n).astype(np.float32)
    c_np = np.random.rand(n, n).astype(np.float32)
    a_cp, b_cp, c_cp = cp.array(a_np), cp.array(b_np), cp.array(c_np)
    return _time_mp_np(
        lambda: cp.linalg.multi_dot([a_cp, b_cp, c_cp]).get(),
        lambda: np.linalg.multi_dot([a_np, b_np, c_np]),
    )


def bench_cross(size):
    a_np = np.random.rand(size // 3, 3).astype(np.float32)
    b_np = np.random.rand(size // 3, 3).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.cross(a_cp, b_cp).get(), lambda: np.cross(a_np, b_np))


def bench_vecdot(size):
    n = size // 3
    a_np = np.random.rand(n, 3).astype(np.float32)
    b_np = np.random.rand(n, 3).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.vecdot(a_cp, b_cp).get(), lambda: np.vecdot(a_np, b_np))


def bench_matvec(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.matvec(a_cp, b_cp).get(), lambda: np.matvec(a_np, b_np))


def bench_vecmat(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n).astype(np.float32)
    b_np = np.random.rand(n, n).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.vecmat(a_cp, b_cp).get(), lambda: np.vecmat(a_np, b_np))


def bench_matrix_transpose(size):
    n = int(size ** 0.5)
    a_np = np.random.rand(n, n).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.matrix_transpose(a_cp).get(), lambda: np.matrix_transpose(a_np))


# ===================================================================
# FFT extra
# ===================================================================
def bench_hfft(size):
    a_np = np.random.rand(size).astype(np.complex64)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.fft.hfft(a_cp).get(), lambda: np.fft.hfft(a_np))


def bench_ihfft(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.fft.ihfft(a_cp).get(), lambda: np.fft.ihfft(a_np))


def bench_rfftfreq(size):
    return _time_mp_np(
        lambda: cp.fft.rfftfreq(size).get(),
        lambda: np.fft.rfftfreq(size),
    )


def bench_fftfreq(size):
    return _time_mp_np(
        lambda: cp.fft.fftfreq(size).get(),
        lambda: np.fft.fftfreq(size),
    )


def bench_fftshift(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.fft.fftshift(a_cp).get(), lambda: np.fft.fftshift(a_np))


def bench_ifftshift(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.fft.ifftshift(a_cp).get(), lambda: np.fft.ifftshift(a_np))


# ===================================================================
# Sorting extra
# ===================================================================
def bench_lexsort(size):
    a_np = np.random.rand(size).astype(np.float32)
    b_np = np.random.rand(size).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(
        lambda: cp.lexsort((a_cp, b_cp)).get(),
        lambda: np.lexsort((a_np, b_np)),
    )


def bench_argpartition(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    kth = size // 2
    return _time_mp_np(
        lambda: cp.argpartition(a_cp, kth).get(),
        lambda: np.argpartition(a_np, kth),
    )


def bench_sort_complex(size):
    a_np = (np.random.rand(size) + 1j * np.random.rand(size)).astype(np.complex64)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.sort_complex(a_cp).get(), lambda: np.sort_complex(a_np))


# ===================================================================
# Misc
# ===================================================================
def bench_clip(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.clip(a_cp, 0.2, 0.8).get(), lambda: np.clip(a_np, 0.2, 0.8))


def bench_nan_to_num(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_np[np.random.rand(size) < 0.1] = np.nan
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.nan_to_num(a_cp).get(), lambda: np.nan_to_num(a_np))


def bench_einsum(size):
    n = int(size ** 0.5)
    n = min(n, 500)
    a_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n, n).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(
        lambda: cp.einsum('ij,jk->ik', a_cp, b_cp).get(),
        lambda: np.einsum('ij,jk->ik', a_np, b_np),
    )


def bench_kron(size):
    n = min(int(size ** 0.25), 30)
    a_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n, n).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.kron(a_cp, b_cp).get(), lambda: np.kron(a_np, b_np))


def bench_vdot(size):
    a_np = np.random.rand(size).astype(np.float32)
    b_np = np.random.rand(size).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.vdot(a_cp, b_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.vdot(a_np, b_np))


def bench_inner(size):
    a_np = np.random.rand(size).astype(np.float32)
    b_np = np.random.rand(size).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.inner(a_cp, b_cp)
        if hasattr(r, 'get'): r.get()
    return _time_mp_np(mp, lambda: np.inner(a_np, b_np))


def bench_outer(size):
    n = min(int(size ** 0.5), 1000)
    a_np = np.random.rand(n).astype(np.float32)
    b_np = np.random.rand(n).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.outer(a_cp, b_cp).get(), lambda: np.outer(a_np, b_np))


def bench_tensordot(size):
    n = int(size ** 0.5)
    n = min(n, 500)
    a_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n, n).astype(np.float32)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(
        lambda: cp.tensordot(a_cp, b_cp, axes=1).get(),
        lambda: np.tensordot(a_np, b_np, axes=1),
    )


def bench_cumulative_sum(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.cumulative_sum(a_cp).get(),
        lambda: np.cumulative_sum(a_np),
    )


def bench_cumulative_prod(size):
    a_np = np.random.rand(size).astype(np.float32) + 0.5
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: cp.cumulative_prod(a_cp).get(),
        lambda: np.cumulative_prod(a_np),
    )


# ===================================================================
# Poly extra
# ===================================================================
def bench_poly(size):
    n = min(size, 50)
    a_np = np.random.rand(n).astype(np.float64)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.poly(a_cp).get(), lambda: np.poly(a_np))


def bench_polyadd(size):
    a_np = np.random.rand(min(size, 100)).astype(np.float64)
    b_np = np.random.rand(min(size, 100)).astype(np.float64)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.polyadd(a_cp, b_cp).get(), lambda: np.polyadd(a_np, b_np))


def bench_polyder(size):
    a_np = np.random.rand(min(size, 100)).astype(np.float64)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.polyder(a_cp).get(), lambda: np.polyder(a_np))


def bench_polydiv(size):
    a_np = np.random.rand(min(size, 100)).astype(np.float64)
    b_np = np.random.rand(min(size, 50)).astype(np.float64)
    b_np[0] = 1.0
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    def mp():
        r = cp.polydiv(a_cp, b_cp)
        r[0].get()
    def np_run():
        np.polydiv(a_np, b_np)
    return _time_mp_np(mp, np_run)


def bench_polyint(size):
    a_np = np.random.rand(min(size, 100)).astype(np.float64)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.polyint(a_cp).get(), lambda: np.polyint(a_np))


def bench_polymul(size):
    a_np = np.random.rand(min(size, 100)).astype(np.float64)
    b_np = np.random.rand(min(size, 100)).astype(np.float64)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.polymul(a_cp, b_cp).get(), lambda: np.polymul(a_np, b_np))


def bench_polysub(size):
    a_np = np.random.rand(min(size, 100)).astype(np.float64)
    b_np = np.random.rand(min(size, 100)).astype(np.float64)
    a_cp, b_cp = cp.array(a_np), cp.array(b_np)
    return _time_mp_np(lambda: cp.polysub(a_cp, b_cp).get(), lambda: np.polysub(a_np, b_np))


# ===================================================================
# Misc extra
# ===================================================================
def bench_frombuffer(size):
    buf = np.random.rand(size).astype(np.float32).tobytes()
    return _time_mp_np(
        lambda: cp.frombuffer(buf, dtype=np.float32).get(),
        lambda: np.frombuffer(buf, dtype=np.float32),
    )


def bench_fromiter(size):
    n = min(size, 10000)
    data = list(range(n))
    return _time_mp_np(
        lambda: cp.fromiter(data, dtype=np.float32, count=n).get(),
        lambda: np.fromiter(data, dtype=np.float32, count=n),
    )


def bench_fromstring(size):
    n = min(size, 1000)
    s = ' '.join(str(float(i)) for i in range(n))
    return _time_mp_np(
        lambda: cp.fromstring(s, dtype=np.float32, sep=' ').get(),
        lambda: np.fromstring(s, dtype=np.float32, sep=' '),
    )


def bench_tril_indices_from(size):
    n = int(size ** 0.5)
    n = min(n, 300)
    a_np = np.zeros((n, n), dtype=np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.tril_indices_from(a_cp)
        r[0].get()
    def np_run():
        np.tril_indices_from(a_np)
    return _time_mp_np(mp, np_run)


def bench_triu_indices_from(size):
    n = int(size ** 0.5)
    n = min(n, 300)
    a_np = np.zeros((n, n), dtype=np.float32)
    a_cp = cp.array(a_np)
    def mp():
        r = cp.triu_indices_from(a_cp)
        r[0].get()
    def np_run():
        np.triu_indices_from(a_np)
    return _time_mp_np(mp, np_run)


def bench_piecewise(size):
    a_np = np.random.rand(size).astype(np.float32) * 2 - 1
    a_cp = cp.array(a_np)
    conds_np = [a_np < 0, a_np >= 0]
    conds_cp = [a_cp < 0, a_cp >= 0]
    funcs = [lambda x: x * 2, lambda x: x * 3]
    return _time_mp_np(
        lambda: cp.piecewise(a_cp, conds_cp, funcs).get(),
        lambda: np.piecewise(a_np, conds_np, funcs),
    )


def bench_isnat(size):
    a_np = np.array([np.datetime64('NaT')] * size)
    a_cp = cp.array(a_np)
    return _time_mp_np(lambda: cp.isnat(a_cp).get(), lambda: np.isnat(a_np))


# ===================================================================
# BENCHMARKS registry
# ===================================================================

BENCHMARKS = [
    # --- Trig (15) ---
    {"name": "arcsin", "category": "trig", "func": bench_arcsin, "sizes": ALL_SIZES},
    {"name": "arccos", "category": "trig", "func": bench_arccos, "sizes": ALL_SIZES},
    {"name": "arctan", "category": "trig", "func": bench_arctan, "sizes": ALL_SIZES},
    {"name": "arctan2", "category": "trig", "func": bench_arctan2, "sizes": ALL_SIZES},
    {"name": "sinh", "category": "trig", "func": bench_sinh, "sizes": ALL_SIZES},
    {"name": "cosh", "category": "trig", "func": bench_cosh, "sizes": ALL_SIZES},
    {"name": "tanh", "category": "trig", "func": bench_tanh, "sizes": ALL_SIZES},
    {"name": "arcsinh", "category": "trig", "func": bench_arcsinh, "sizes": ALL_SIZES},
    {"name": "arccosh", "category": "trig", "func": bench_arccosh, "sizes": ALL_SIZES},
    {"name": "arctanh", "category": "trig", "func": bench_arctanh, "sizes": ALL_SIZES},
    {"name": "degrees", "category": "trig", "func": bench_degrees, "sizes": ALL_SIZES},
    {"name": "radians", "category": "trig", "func": bench_radians, "sizes": ALL_SIZES},
    {"name": "deg2rad", "category": "trig", "func": bench_deg2rad, "sizes": ALL_SIZES},
    {"name": "rad2deg", "category": "trig", "func": bench_rad2deg, "sizes": ALL_SIZES},
    {"name": "hypot", "category": "trig", "func": bench_hypot, "sizes": ALL_SIZES},

    # --- Ufuncs extra (33) ---
    {"name": "exp2", "category": "ufuncs_extra", "func": bench_exp2, "sizes": ALL_SIZES},
    {"name": "expm1", "category": "ufuncs_extra", "func": bench_expm1, "sizes": ALL_SIZES},
    {"name": "log1p", "category": "ufuncs_extra", "func": bench_log1p, "sizes": ALL_SIZES},
    {"name": "log2", "category": "ufuncs_extra", "func": bench_log2, "sizes": ALL_SIZES},
    {"name": "log10", "category": "ufuncs_extra", "func": bench_log10, "sizes": ALL_SIZES},
    {"name": "cbrt", "category": "ufuncs_extra", "func": bench_cbrt, "sizes": ALL_SIZES},
    {"name": "reciprocal", "category": "ufuncs_extra", "func": bench_reciprocal, "sizes": ALL_SIZES},
    {"name": "rint", "category": "ufuncs_extra", "func": bench_rint, "sizes": ALL_SIZES},
    {"name": "trunc", "category": "ufuncs_extra", "func": bench_trunc, "sizes": ALL_SIZES},
    {"name": "sign", "category": "ufuncs_extra", "func": bench_sign, "sizes": ALL_SIZES},
    {"name": "floor", "category": "ufuncs_extra", "func": bench_floor, "sizes": ALL_SIZES},
    {"name": "ceil", "category": "ufuncs_extra", "func": bench_ceil, "sizes": ALL_SIZES},
    {"name": "square", "category": "ufuncs_extra", "func": bench_square, "sizes": ALL_SIZES},
    {"name": "negative", "category": "ufuncs_extra", "func": bench_negative, "sizes": ALL_SIZES},
    {"name": "positive", "category": "ufuncs_extra", "func": bench_positive, "sizes": ALL_SIZES},
    {"name": "absolute", "category": "ufuncs_extra", "func": bench_absolute, "sizes": ALL_SIZES},
    {"name": "fabs", "category": "ufuncs_extra", "func": bench_fabs, "sizes": ALL_SIZES},
    {"name": "float_power", "category": "ufuncs_extra", "func": bench_float_power, "sizes": ALL_SIZES},
    {"name": "copysign", "category": "ufuncs_extra", "func": bench_copysign, "sizes": ALL_SIZES},
    {"name": "nextafter", "category": "ufuncs_extra", "func": bench_nextafter, "sizes": ALL_SIZES},
    {"name": "heaviside", "category": "ufuncs_extra", "func": bench_heaviside, "sizes": ALL_SIZES},
    {"name": "signbit", "category": "ufuncs_extra", "func": bench_signbit, "sizes": ALL_SIZES},
    {"name": "modf", "category": "ufuncs_extra", "func": bench_modf, "sizes": ALL_SIZES},
    {"name": "ldexp", "category": "ufuncs_extra", "func": bench_ldexp, "sizes": ALL_SIZES},
    {"name": "frexp", "category": "ufuncs_extra", "func": bench_frexp, "sizes": ALL_SIZES},
    {"name": "fmax", "category": "ufuncs_extra", "func": bench_fmax, "sizes": ALL_SIZES},
    {"name": "fmin", "category": "ufuncs_extra", "func": bench_fmin, "sizes": ALL_SIZES},
    {"name": "fmod", "category": "ufuncs_extra", "func": bench_fmod, "sizes": ALL_SIZES},
    {"name": "remainder", "category": "ufuncs_extra", "func": bench_remainder, "sizes": ALL_SIZES},
    {"name": "true_divide", "category": "ufuncs_extra", "func": bench_true_divide, "sizes": ALL_SIZES},
    {"name": "logaddexp", "category": "ufuncs_extra", "func": bench_logaddexp, "sizes": ALL_SIZES},
    {"name": "logaddexp2", "category": "ufuncs_extra", "func": bench_logaddexp2, "sizes": ALL_SIZES},
    {"name": "maximum", "category": "ufuncs_extra", "func": bench_maximum, "sizes": ALL_SIZES},
    {"name": "minimum", "category": "ufuncs_extra", "func": bench_minimum, "sizes": ALL_SIZES},

    # --- Bitwise (9) ---
    {"name": "bitwise_and", "category": "bitwise", "func": bench_bitwise_and, "sizes": ALL_SIZES},
    {"name": "bitwise_or", "category": "bitwise", "func": bench_bitwise_or, "sizes": ALL_SIZES},
    {"name": "bitwise_xor", "category": "bitwise", "func": bench_bitwise_xor, "sizes": ALL_SIZES},
    {"name": "invert", "category": "bitwise", "func": bench_invert, "sizes": ALL_SIZES},
    {"name": "left_shift", "category": "bitwise", "func": bench_left_shift, "sizes": ALL_SIZES},
    {"name": "right_shift", "category": "bitwise", "func": bench_right_shift, "sizes": ALL_SIZES},
    {"name": "gcd", "category": "bitwise", "func": bench_gcd, "sizes": ALL_SIZES},
    {"name": "lcm", "category": "bitwise", "func": bench_lcm, "sizes": ALL_SIZES},
    {"name": "bitwise_count", "category": "bitwise", "func": bench_bitwise_count, "sizes": ALL_SIZES},

    # --- Logic (16) ---
    {"name": "logical_and", "category": "logic", "func": bench_logical_and, "sizes": ALL_SIZES},
    {"name": "logical_or", "category": "logic", "func": bench_logical_or, "sizes": ALL_SIZES},
    {"name": "logical_not", "category": "logic", "func": bench_logical_not, "sizes": ALL_SIZES},
    {"name": "logical_xor", "category": "logic", "func": bench_logical_xor, "sizes": ALL_SIZES},
    {"name": "isnan", "category": "logic", "func": bench_isnan, "sizes": ALL_SIZES},
    {"name": "isinf", "category": "logic", "func": bench_isinf, "sizes": ALL_SIZES},
    {"name": "isfinite", "category": "logic", "func": bench_isfinite, "sizes": ALL_SIZES},
    {"name": "isclose", "category": "logic", "func": bench_isclose, "sizes": ALL_SIZES},
    {"name": "allclose", "category": "logic", "func": bench_allclose, "sizes": ALL_SIZES},
    {"name": "array_equal", "category": "logic", "func": bench_array_equal, "sizes": ALL_SIZES},
    {"name": "isneginf", "category": "logic", "func": bench_isneginf, "sizes": ALL_SIZES},
    {"name": "isposinf", "category": "logic", "func": bench_isposinf, "sizes": ALL_SIZES},
    {"name": "iscomplex", "category": "logic", "func": bench_iscomplex, "sizes": ALL_SIZES},
    {"name": "isreal", "category": "logic", "func": bench_isreal, "sizes": ALL_SIZES},
    {"name": "greater_equal", "category": "logic", "func": bench_greater_equal, "sizes": ALL_SIZES},
    {"name": "less_equal", "category": "logic", "func": bench_less_equal, "sizes": ALL_SIZES},

    # --- Window (5) ---
    {"name": "bartlett", "category": "window", "func": bench_bartlett, "sizes": ALL_SIZES},
    {"name": "blackman", "category": "window", "func": bench_blackman, "sizes": ALL_SIZES},
    {"name": "hamming", "category": "window", "func": bench_hamming, "sizes": ALL_SIZES},
    {"name": "hanning", "category": "window", "func": bench_hanning, "sizes": ALL_SIZES},
    {"name": "kaiser", "category": "window", "func": bench_kaiser, "sizes": ALL_SIZES},

    # --- Complex ops (5) ---
    {"name": "angle", "category": "complex_ops", "func": bench_angle, "sizes": ALL_SIZES},
    {"name": "real", "category": "complex_ops", "func": bench_real, "sizes": ALL_SIZES},
    {"name": "imag", "category": "complex_ops", "func": bench_imag, "sizes": ALL_SIZES},
    {"name": "conj", "category": "complex_ops", "func": bench_conj, "sizes": ALL_SIZES},
    {"name": "real_if_close", "category": "complex_ops", "func": bench_real_if_close, "sizes": ALL_SIZES},

    # --- Manipulation extra (28) ---
    {"name": "pad", "category": "manipulation_extra", "func": bench_pad, "sizes": ALL_SIZES},
    {"name": "delete", "category": "manipulation_extra", "func": bench_delete, "sizes": ALL_SIZES},
    {"name": "insert", "category": "manipulation_extra", "func": bench_insert, "sizes": ALL_SIZES},
    {"name": "block", "category": "manipulation_extra", "func": bench_block, "sizes": ALL_SIZES},
    {"name": "broadcast_to", "category": "manipulation_extra", "func": bench_broadcast_to, "sizes": ALL_SIZES},
    {"name": "moveaxis", "category": "manipulation_extra", "func": bench_moveaxis, "sizes": ALL_SIZES},
    {"name": "swapaxes", "category": "manipulation_extra", "func": bench_swapaxes, "sizes": ALL_SIZES},
    {"name": "rollaxis", "category": "manipulation_extra", "func": bench_rollaxis, "sizes": ALL_SIZES},
    {"name": "atleast_1d", "category": "manipulation_extra", "func": bench_atleast_1d, "sizes": ALL_SIZES},
    {"name": "atleast_2d", "category": "manipulation_extra", "func": bench_atleast_2d, "sizes": ALL_SIZES},
    {"name": "atleast_3d", "category": "manipulation_extra", "func": bench_atleast_3d, "sizes": ALL_SIZES},
    {"name": "dstack", "category": "manipulation_extra", "func": bench_dstack, "sizes": ALL_SIZES},
    {"name": "column_stack", "category": "manipulation_extra", "func": bench_column_stack, "sizes": ALL_SIZES},
    {"name": "dsplit", "category": "manipulation_extra", "func": bench_dsplit, "sizes": ALL_SIZES},
    {"name": "hsplit", "category": "manipulation_extra", "func": bench_hsplit, "sizes": ALL_SIZES},
    {"name": "vsplit", "category": "manipulation_extra", "func": bench_vsplit, "sizes": ALL_SIZES},
    {"name": "fliplr", "category": "manipulation_extra", "func": bench_fliplr, "sizes": ALL_SIZES},
    {"name": "flipud", "category": "manipulation_extra", "func": bench_flipud, "sizes": ALL_SIZES},
    {"name": "rot90", "category": "manipulation_extra", "func": bench_rot90, "sizes": ALL_SIZES},
    {"name": "copyto", "category": "manipulation_extra", "func": bench_copyto, "sizes": ALL_SIZES},
    {"name": "trim_zeros", "category": "manipulation_extra", "func": bench_trim_zeros, "sizes": ALL_SIZES},
    {"name": "resize", "category": "manipulation_extra", "func": bench_resize, "sizes": ALL_SIZES},
    {"name": "append", "category": "manipulation_extra", "func": bench_append, "sizes": ALL_SIZES},
    {"name": "broadcast_arrays", "category": "manipulation_extra", "func": bench_broadcast_arrays, "sizes": ALL_SIZES},
    {"name": "asfortranarray", "category": "manipulation_extra", "func": bench_asfortranarray, "sizes": ALL_SIZES},
    {"name": "ascontiguousarray", "category": "manipulation_extra", "func": bench_ascontiguousarray, "sizes": ALL_SIZES},
    {"name": "copy", "category": "manipulation_extra", "func": bench_copy, "sizes": ALL_SIZES},

    # --- Creation extra (18) ---
    {"name": "fromfunction", "category": "creation_extra", "func": bench_fromfunction, "sizes": ALL_SIZES},
    {"name": "diagflat", "category": "creation_extra", "func": bench_diagflat, "sizes": ALL_SIZES},
    {"name": "vander", "category": "creation_extra", "func": bench_vander, "sizes": ["small", "medium"]},
    {"name": "geomspace", "category": "creation_extra", "func": bench_geomspace, "sizes": ALL_SIZES},
    {"name": "asarray", "category": "creation_extra", "func": bench_asarray, "sizes": ALL_SIZES},
    {"name": "asanyarray", "category": "creation_extra", "func": bench_asanyarray, "sizes": ALL_SIZES},
    {"name": "asarray_chkfinite", "category": "creation_extra", "func": bench_asarray_chkfinite, "sizes": ALL_SIZES},
    {"name": "identity", "category": "creation_extra", "func": bench_identity, "sizes": ALL_SIZES},
    {"name": "tri", "category": "creation_extra", "func": bench_tri, "sizes": ALL_SIZES},
    {"name": "triu", "category": "creation_extra", "func": bench_triu, "sizes": ALL_SIZES},
    {"name": "tril", "category": "creation_extra", "func": bench_tril, "sizes": ALL_SIZES},
    {"name": "diag", "category": "creation_extra", "func": bench_diag, "sizes": ALL_SIZES},
    {"name": "indices", "category": "creation_extra", "func": bench_indices, "sizes": ALL_SIZES},
    {"name": "meshgrid", "category": "creation_extra", "func": bench_meshgrid, "sizes": ALL_SIZES},
    {"name": "logspace", "category": "creation_extra", "func": bench_logspace, "sizes": ALL_SIZES},
    {"name": "around", "category": "creation_extra", "func": bench_around, "sizes": ALL_SIZES},
    {"name": "frombuffer", "category": "creation_extra", "func": bench_frombuffer, "sizes": ALL_SIZES},
    {"name": "fromstring", "category": "creation_extra", "func": bench_fromstring, "sizes": ["small"]},

    # --- Reduction extra (8) ---
    {"name": "count_nonzero", "category": "reduction_extra", "func": bench_count_nonzero, "sizes": ALL_SIZES},
    {"name": "amax", "category": "reduction_extra", "func": bench_amax, "sizes": ALL_SIZES},
    {"name": "amin", "category": "reduction_extra", "func": bench_amin, "sizes": ALL_SIZES},
    {"name": "ptp", "category": "reduction_extra", "func": bench_ptp, "sizes": ALL_SIZES},
    {"name": "average", "category": "reduction_extra", "func": bench_average, "sizes": ALL_SIZES},
    {"name": "percentile", "category": "reduction_extra", "func": bench_percentile, "sizes": ALL_SIZES},
    {"name": "quantile", "category": "reduction_extra", "func": bench_quantile, "sizes": ALL_SIZES},
    {"name": "median", "category": "reduction_extra", "func": bench_median, "sizes": ALL_SIZES},
    {"name": "divmod", "category": "reduction_extra", "func": bench_divmod_fn, "sizes": ALL_SIZES},

    # --- NaN extra (9) ---
    {"name": "nanprod", "category": "nan_extra", "func": bench_nanprod, "sizes": ALL_SIZES},
    {"name": "nancumsum", "category": "nan_extra", "func": bench_nancumsum, "sizes": ALL_SIZES},
    {"name": "nancumprod", "category": "nan_extra", "func": bench_nancumprod, "sizes": ALL_SIZES},
    {"name": "nanargmax", "category": "nan_extra", "func": bench_nanargmax, "sizes": ALL_SIZES},
    {"name": "nanargmin", "category": "nan_extra", "func": bench_nanargmin, "sizes": ALL_SIZES},
    {"name": "nanvar", "category": "nan_extra", "func": bench_nanvar, "sizes": ALL_SIZES},
    {"name": "nanmedian", "category": "nan_extra", "func": bench_nanmedian, "sizes": ALL_SIZES},
    {"name": "nanpercentile", "category": "nan_extra", "func": bench_nanpercentile, "sizes": ALL_SIZES},
    {"name": "nanquantile", "category": "nan_extra", "func": bench_nanquantile, "sizes": ALL_SIZES},

    # --- Indexing extra (17) ---
    {"name": "fill_diagonal", "category": "indexing_extra", "func": bench_fill_diagonal, "sizes": ALL_SIZES},
    {"name": "diag_indices", "category": "indexing_extra", "func": bench_diag_indices, "sizes": ALL_SIZES},
    {"name": "diag_indices_from", "category": "indexing_extra", "func": bench_diag_indices_from, "sizes": ALL_SIZES},
    {"name": "tril_indices", "category": "indexing_extra", "func": bench_tril_indices, "sizes": ALL_SIZES},
    {"name": "triu_indices", "category": "indexing_extra", "func": bench_triu_indices, "sizes": ALL_SIZES},
    {"name": "tril_indices_from", "category": "indexing_extra", "func": bench_tril_indices_from, "sizes": ALL_SIZES},
    {"name": "triu_indices_from", "category": "indexing_extra", "func": bench_triu_indices_from, "sizes": ALL_SIZES},
    {"name": "ravel_multi_index", "category": "indexing_extra", "func": bench_ravel_multi_index, "sizes": ALL_SIZES},
    {"name": "unravel_index", "category": "indexing_extra", "func": bench_unravel_index, "sizes": ALL_SIZES},
    {"name": "flatnonzero", "category": "indexing_extra", "func": bench_flatnonzero, "sizes": ALL_SIZES},
    {"name": "choose", "category": "indexing_extra", "func": bench_choose, "sizes": ["small", "medium"]},
    {"name": "compress", "category": "indexing_extra", "func": bench_compress, "sizes": ALL_SIZES},
    {"name": "select", "category": "indexing_extra", "func": bench_select, "sizes": ALL_SIZES},
    {"name": "extract", "category": "indexing_extra", "func": bench_extract, "sizes": ALL_SIZES},
    {"name": "putmask", "category": "indexing_extra", "func": bench_putmask, "sizes": ALL_SIZES},
    {"name": "place", "category": "indexing_extra", "func": bench_place, "sizes": ALL_SIZES},
    {"name": "take_along_axis", "category": "indexing_extra", "func": bench_take_along_axis, "sizes": ALL_SIZES},
    {"name": "put_along_axis", "category": "indexing_extra", "func": bench_put_along_axis, "sizes": ALL_SIZES},
    {"name": "ix_", "category": "indexing_extra", "func": bench_ix_, "sizes": ALL_SIZES},
    {"name": "mask_indices", "category": "indexing_extra", "func": bench_mask_indices, "sizes": ALL_SIZES},

    # --- Set extra (1) ---
    {"name": "setxor1d", "category": "set_extra", "func": bench_setxor1d, "sizes": ALL_SIZES},

    # --- Math ext (11) ---
    {"name": "sinc", "category": "math_ext", "func": bench_sinc, "sizes": ALL_SIZES},
    {"name": "convolve", "category": "math_ext", "func": bench_convolve, "sizes": ["small", "medium"]},
    {"name": "interp", "category": "math_ext", "func": bench_interp, "sizes": ALL_SIZES},
    {"name": "gradient", "category": "math_ext", "func": bench_gradient, "sizes": ALL_SIZES},
    {"name": "trapezoid", "category": "math_ext", "func": bench_trapezoid, "sizes": ALL_SIZES},
    {"name": "i0", "category": "math_ext", "func": bench_i0, "sizes": ALL_SIZES},
    {"name": "fix", "category": "math_ext", "func": bench_fix, "sizes": ALL_SIZES},
    {"name": "unwrap", "category": "math_ext", "func": bench_unwrap, "sizes": ALL_SIZES},
    {"name": "spacing", "category": "math_ext", "func": bench_spacing, "sizes": ALL_SIZES},
    {"name": "diff", "category": "math_ext", "func": bench_diff, "sizes": ALL_SIZES},
    {"name": "ediff1d", "category": "math_ext", "func": bench_ediff1d, "sizes": ALL_SIZES},

    # --- Stats (10) ---
    {"name": "histogram", "category": "stats", "func": bench_histogram, "sizes": ALL_SIZES},
    {"name": "histogram2d", "category": "stats", "func": bench_histogram2d, "sizes": ALL_SIZES},
    {"name": "histogramdd", "category": "stats", "func": bench_histogramdd, "sizes": ALL_SIZES},
    {"name": "histogram_bin_edges", "category": "stats", "func": bench_histogram_bin_edges, "sizes": ALL_SIZES},
    {"name": "bincount", "category": "stats", "func": bench_bincount, "sizes": ALL_SIZES},
    {"name": "digitize", "category": "stats", "func": bench_digitize, "sizes": ALL_SIZES},
    {"name": "corrcoef", "category": "stats", "func": bench_corrcoef, "sizes": ["small", "medium"]},
    {"name": "correlate", "category": "stats", "func": bench_correlate, "sizes": ["small", "medium"]},
    {"name": "cov", "category": "stats", "func": bench_cov, "sizes": ["small", "medium"]},

    # --- Linalg extra (13) ---
    {"name": "lstsq", "category": "linalg_extra", "func": bench_lstsq, "sizes": ["small", "medium"]},
    {"name": "pinv", "category": "linalg_extra", "func": bench_pinv, "sizes": ["small", "medium"]},
    {"name": "matrix_rank", "category": "linalg_extra", "func": bench_matrix_rank, "sizes": ["small", "medium"]},
    {"name": "matrix_power", "category": "linalg_extra", "func": bench_matrix_power, "sizes": ["small", "medium"]},
    {"name": "eigvals", "category": "linalg_extra", "func": bench_eigvals, "sizes": ["small", "medium"]},
    {"name": "cond", "category": "linalg_extra", "func": bench_cond, "sizes": ["small", "medium"]},
    {"name": "multi_dot", "category": "linalg_extra", "func": bench_multi_dot, "sizes": ["small", "medium"]},
    {"name": "cross", "category": "linalg_extra", "func": bench_cross, "sizes": ALL_SIZES},
    {"name": "vecdot", "category": "linalg_extra", "func": bench_vecdot, "sizes": ALL_SIZES},
    {"name": "matvec", "category": "linalg_extra", "func": bench_matvec, "sizes": ["small", "medium"]},
    {"name": "vecmat", "category": "linalg_extra", "func": bench_vecmat, "sizes": ["small", "medium"]},
    {"name": "matrix_transpose", "category": "linalg_extra", "func": bench_matrix_transpose, "sizes": ALL_SIZES},

    # --- FFT extra (6) ---
    {"name": "hfft", "category": "fft_extra", "func": bench_hfft, "sizes": ALL_SIZES},
    {"name": "ihfft", "category": "fft_extra", "func": bench_ihfft, "sizes": ALL_SIZES},
    {"name": "rfftfreq", "category": "fft_extra", "func": bench_rfftfreq, "sizes": ALL_SIZES},
    {"name": "fftfreq", "category": "fft_extra", "func": bench_fftfreq, "sizes": ALL_SIZES},
    {"name": "fftshift", "category": "fft_extra", "func": bench_fftshift, "sizes": ALL_SIZES},
    {"name": "ifftshift", "category": "fft_extra", "func": bench_ifftshift, "sizes": ALL_SIZES},

    # --- Sorting extra (3) ---
    {"name": "lexsort", "category": "sorting_extra", "func": bench_lexsort, "sizes": ALL_SIZES},
    {"name": "argpartition", "category": "sorting_extra", "func": bench_argpartition, "sizes": ALL_SIZES},
    {"name": "sort_complex", "category": "sorting_extra", "func": bench_sort_complex, "sizes": ALL_SIZES},

    # --- Misc (14) ---
    {"name": "clip", "category": "misc", "func": bench_clip, "sizes": ALL_SIZES},
    {"name": "nan_to_num", "category": "misc", "func": bench_nan_to_num, "sizes": ALL_SIZES},
    {"name": "einsum", "category": "misc", "func": bench_einsum, "sizes": ["small", "medium"]},
    {"name": "kron", "category": "misc", "func": bench_kron, "sizes": ALL_SIZES},
    {"name": "vdot", "category": "misc", "func": bench_vdot, "sizes": ALL_SIZES},
    {"name": "inner", "category": "misc", "func": bench_inner, "sizes": ALL_SIZES},
    {"name": "outer", "category": "misc", "func": bench_outer, "sizes": ALL_SIZES},
    {"name": "tensordot", "category": "misc", "func": bench_tensordot, "sizes": ["small", "medium"]},
    {"name": "cumulative_sum", "category": "misc", "func": bench_cumulative_sum, "sizes": ALL_SIZES},
    {"name": "cumulative_prod", "category": "misc", "func": bench_cumulative_prod, "sizes": ALL_SIZES},
    {"name": "piecewise", "category": "misc", "func": bench_piecewise, "sizes": ALL_SIZES},

    # --- Poly extra (7) ---
    {"name": "poly", "category": "poly_extra", "func": bench_poly, "sizes": ["small"]},
    {"name": "polyadd", "category": "poly_extra", "func": bench_polyadd, "sizes": ["small"]},
    {"name": "polyder", "category": "poly_extra", "func": bench_polyder, "sizes": ["small"]},
    {"name": "polydiv", "category": "poly_extra", "func": bench_polydiv, "sizes": ["small"]},
    {"name": "polyint", "category": "poly_extra", "func": bench_polyint, "sizes": ["small"]},
    {"name": "polymul", "category": "poly_extra", "func": bench_polymul, "sizes": ["small"]},
    {"name": "polysub", "category": "poly_extra", "func": bench_polysub, "sizes": ["small"]},
]
