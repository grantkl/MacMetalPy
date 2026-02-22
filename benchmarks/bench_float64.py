"""Float64 benchmark functions for macmetalpy vs NumPy.

Measures overhead of macmetalpy float64 CPU fallback vs raw NumPy float64.
"""

import math
import time

import numpy as np

import macmetalpy as cp

SIZE_MAP = {"small": 1_000, "medium": 100_000, "large": 1_000_000}
ALL_SIZES = ["small", "medium", "large", "xlarge"]

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


def _resolve(module, name):
    """Resolve a dotted name like 'linalg.norm' on a module."""
    obj = module
    for part in name.split('.'):
        obj = getattr(obj, part)
    return obj


class _UnaryF64:
    """Picklable callable for benchmarking unary functions (float64)."""
    def __init__(self, cp_name, np_name, offset=0.1):
        self.cp_name = cp_name
        self.np_name = np_name
        self.offset = offset

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = np.random.rand(size).astype(np.float64) + self.offset
        a_cp = cp.array(a_np, dtype=np.float64)
        return _time_mp_np(lambda: cp_f(a_cp).get(), lambda: np_f(a_np))


class _BinaryF64:
    """Picklable callable for benchmarking binary functions (float64)."""
    def __init__(self, cp_name, np_name, offset=0.1):
        self.cp_name = cp_name
        self.np_name = np_name
        self.offset = offset

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = np.random.rand(size).astype(np.float64) + self.offset
        b_np = np.random.rand(size).astype(np.float64) + self.offset
        a_cp, b_cp = cp.array(a_np, dtype=np.float64), cp.array(b_np, dtype=np.float64)
        return _time_mp_np(lambda: cp_f(a_cp, b_cp).get(), lambda: np_f(a_np, b_np))


class _ReductionF64:
    """Picklable callable for benchmarking reduction functions (float64)."""
    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_f = _resolve(cp, self.cp_name)
        np_f = _resolve(np, self.np_name)
        a_np = np.random.rand(size).astype(np.float64)
        a_cp = cp.array(a_np, dtype=np.float64)
        def mp():
            r = cp_f(a_cp)
            if hasattr(r, 'get'): r.get()
        def np_run():
            np_f(a_np)
        return _time_mp_np(mp, np_run)


# --- Arithmetic f64 (6) ---
bench_add_f64 = _BinaryF64("add", "add")
bench_subtract_f64 = _BinaryF64("subtract", "subtract")
bench_multiply_f64 = _BinaryF64("multiply", "multiply")
bench_divide_f64 = _BinaryF64("divide", "divide")
bench_power_f64 = _BinaryF64("power", "power")
bench_floor_divide_f64 = _BinaryF64("floor_divide", "floor_divide")

# --- Trig f64 (6) ---
bench_sqrt_f64 = _UnaryF64("sqrt", "sqrt")
bench_sin_f64 = _UnaryF64("sin", "sin")
bench_cos_f64 = _UnaryF64("cos", "cos")
bench_exp_f64 = _UnaryF64("exp", "exp")
bench_log_f64 = _UnaryF64("log", "log")
bench_tanh_f64 = _UnaryF64("tanh", "tanh")

# --- Reduction f64 (8) ---
bench_sum_f64 = _ReductionF64("sum", "sum")
bench_mean_f64 = _ReductionF64("mean", "mean")
bench_max_f64 = _ReductionF64("max", "max")
bench_min_f64 = _ReductionF64("min", "min")
bench_std_f64 = _ReductionF64("std", "std")
bench_var_f64 = _ReductionF64("var", "var")
bench_prod_f64 = _ReductionF64("prod", "prod")
bench_cumsum_f64 = _ReductionF64("cumsum", "cumsum")

# --- Comparison f64 (4) ---
bench_equal_f64 = _BinaryF64("equal", "equal")
bench_not_equal_f64 = _BinaryF64("not_equal", "not_equal")
bench_less_f64 = _BinaryF64("less", "less")
bench_greater_f64 = _BinaryF64("greater", "greater")


# --- Creation f64 (4) ---
def bench_array_f64(size):
    a_np = np.random.rand(size).astype(np.float64)
    return _time_mp_np(
        lambda: cp.array(a_np, dtype=np.float64),
        lambda: np.array(a_np, dtype=np.float64),
    )


def bench_zeros_f64(size):
    return _time_mp_np(
        lambda: cp.zeros(size, dtype=np.float64),
        lambda: np.zeros(size, dtype=np.float64),
    )


def bench_ones_f64(size):
    return _time_mp_np(
        lambda: cp.ones(size, dtype=np.float64),
        lambda: np.ones(size, dtype=np.float64),
    )


def bench_arange_f64(size):
    return _time_mp_np(
        lambda: cp.arange(size, dtype=np.float64),
        lambda: np.arange(size, dtype=np.float64),
    )


# --- Casting f64 (2) ---
def bench_astype_f64_to_f32(size):
    a_np = np.random.rand(size).astype(np.float64)
    a_cp = cp.array(a_np, dtype=np.float64)
    return _time_mp_np(
        lambda: a_cp.astype(np.float32),
        lambda: a_np.astype(np.float32),
    )


def bench_astype_f32_to_f64(size):
    a_np = np.random.rand(size).astype(np.float32)
    a_cp = cp.array(a_np)
    return _time_mp_np(
        lambda: a_cp.astype(np.float64),
        lambda: a_np.astype(np.float64),
    )


BENCHMARKS = [
    # --- Arithmetic f64 (6) ---
    {"name": "add_f64", "category": "arithmetic_f64", "func": bench_add_f64, "sizes": ALL_SIZES},
    {"name": "subtract_f64", "category": "arithmetic_f64", "func": bench_subtract_f64, "sizes": ALL_SIZES},
    {"name": "multiply_f64", "category": "arithmetic_f64", "func": bench_multiply_f64, "sizes": ALL_SIZES},
    {"name": "divide_f64", "category": "arithmetic_f64", "func": bench_divide_f64, "sizes": ALL_SIZES},
    {"name": "power_f64", "category": "arithmetic_f64", "func": bench_power_f64, "sizes": ALL_SIZES},
    {"name": "floor_divide_f64", "category": "arithmetic_f64", "func": bench_floor_divide_f64, "sizes": ALL_SIZES},
    # --- Trig f64 (6) ---
    {"name": "sqrt_f64", "category": "trig_f64", "func": bench_sqrt_f64, "sizes": ALL_SIZES},
    {"name": "sin_f64", "category": "trig_f64", "func": bench_sin_f64, "sizes": ALL_SIZES},
    {"name": "cos_f64", "category": "trig_f64", "func": bench_cos_f64, "sizes": ALL_SIZES},
    {"name": "exp_f64", "category": "trig_f64", "func": bench_exp_f64, "sizes": ALL_SIZES},
    {"name": "log_f64", "category": "trig_f64", "func": bench_log_f64, "sizes": ALL_SIZES},
    {"name": "tanh_f64", "category": "trig_f64", "func": bench_tanh_f64, "sizes": ALL_SIZES},
    # --- Reduction f64 (8) ---
    {"name": "sum_f64", "category": "reduction_f64", "func": bench_sum_f64, "sizes": ALL_SIZES},
    {"name": "mean_f64", "category": "reduction_f64", "func": bench_mean_f64, "sizes": ALL_SIZES},
    {"name": "max_f64", "category": "reduction_f64", "func": bench_max_f64, "sizes": ALL_SIZES},
    {"name": "min_f64", "category": "reduction_f64", "func": bench_min_f64, "sizes": ALL_SIZES},
    {"name": "std_f64", "category": "reduction_f64", "func": bench_std_f64, "sizes": ALL_SIZES},
    {"name": "var_f64", "category": "reduction_f64", "func": bench_var_f64, "sizes": ALL_SIZES},
    {"name": "prod_f64", "category": "reduction_f64", "func": bench_prod_f64, "sizes": ALL_SIZES},
    {"name": "cumsum_f64", "category": "reduction_f64", "func": bench_cumsum_f64, "sizes": ALL_SIZES},
    # --- Comparison f64 (4) ---
    {"name": "equal_f64", "category": "comparison_f64", "func": bench_equal_f64, "sizes": ALL_SIZES},
    {"name": "not_equal_f64", "category": "comparison_f64", "func": bench_not_equal_f64, "sizes": ALL_SIZES},
    {"name": "less_f64", "category": "comparison_f64", "func": bench_less_f64, "sizes": ALL_SIZES},
    {"name": "greater_f64", "category": "comparison_f64", "func": bench_greater_f64, "sizes": ALL_SIZES},
    # --- Creation f64 (4) ---
    {"name": "array_f64", "category": "creation_f64", "func": bench_array_f64, "sizes": ALL_SIZES},
    {"name": "zeros_f64", "category": "creation_f64", "func": bench_zeros_f64, "sizes": ALL_SIZES},
    {"name": "ones_f64", "category": "creation_f64", "func": bench_ones_f64, "sizes": ALL_SIZES},
    {"name": "arange_f64", "category": "creation_f64", "func": bench_arange_f64, "sizes": ALL_SIZES},
    # --- Casting f64 (2) ---
    {"name": "astype_f64_to_f32", "category": "casting_f64", "func": bench_astype_f64_to_f32, "sizes": ALL_SIZES},
    {"name": "astype_f32_to_f64", "category": "casting_f64", "func": bench_astype_f32_to_f64, "sizes": ALL_SIZES},
]
