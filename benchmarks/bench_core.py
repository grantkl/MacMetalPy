"""Core benchmark functions comparing macmetalpy vs NumPy performance."""

import math
import time

import numpy as np

import macmetalpy as cp

# ---------------------------------------------------------------------------
# Size mapping
# ---------------------------------------------------------------------------
SIZE_MAP = {
    "small": 1_000,
    "medium": 100_000,
    "large": 1_000_000,
}

ALL_SIZES = ["small", "medium", "large", "xlarge"]

# Number of warmup + timed iterations for stable medians
_WARMUP = 2
_REPEATS = 5


def _median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


# ===================================================================
# Creation benchmarks
# ===================================================================

def bench_array(size):
    data_np = np.random.rand(size).astype(np.float32)

    # warmup
    for _ in range(_WARMUP):
        cp.array(data_np).get()

    # macmetalpy
    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        result = cp.array(data_np)
        _ = result.get()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    # numpy (array copy)
    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        _ = np.array(data_np, copy=True)
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


def bench_zeros(size):
    for _ in range(_WARMUP):
        cp.zeros(size, dtype=np.float32).get()

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        result = cp.zeros(size, dtype=np.float32)
        _ = result.get()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        _ = np.zeros(size, dtype=np.float32)
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


def bench_ones(size):
    for _ in range(_WARMUP):
        cp.ones(size, dtype=np.float32).get()

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        result = cp.ones(size, dtype=np.float32)
        _ = result.get()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        _ = np.ones(size, dtype=np.float32)
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


def bench_empty(size):
    for _ in range(_WARMUP):
        cp.empty(size, dtype=np.float32).get()

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        result = cp.empty(size, dtype=np.float32)
        _ = result.get()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        _ = np.empty(size, dtype=np.float32)
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


def bench_arange(size):
    for _ in range(_WARMUP):
        cp.arange(size, dtype=np.float32).get()

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        result = cp.arange(size, dtype=np.float32)
        _ = result.get()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        _ = np.arange(size, dtype=np.float32)
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


def bench_linspace(size):
    for _ in range(_WARMUP):
        cp.linspace(0, 1, size, dtype=np.float32).get()

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        result = cp.linspace(0, 1, size, dtype=np.float32)
        _ = result.get()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        _ = np.linspace(0, 1, size, dtype=np.float32)
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


def bench_eye(size):
    n = int(math.sqrt(size))

    for _ in range(_WARMUP):
        cp.eye(n, dtype=np.float32).get()

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        result = cp.eye(n, dtype=np.float32)
        _ = result.get()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        _ = np.eye(n, dtype=np.float32)
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


def bench_full(size):
    for _ in range(_WARMUP):
        cp.full(size, 3.14, dtype=np.float32).get()

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        result = cp.full(size, 3.14, dtype=np.float32)
        _ = result.get()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        _ = np.full(size, 3.14, dtype=np.float32)
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


def bench_zeros_like(size):
    data_np = np.random.rand(size).astype(np.float32)
    data_cp = cp.array(data_np)

    for _ in range(_WARMUP):
        cp.zeros_like(data_cp).get()

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        result = cp.zeros_like(data_cp)
        _ = result.get()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        _ = np.zeros_like(data_np)
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


def bench_ones_like(size):
    data_np = np.random.rand(size).astype(np.float32)
    data_cp = cp.array(data_np)

    for _ in range(_WARMUP):
        cp.ones_like(data_cp).get()

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        result = cp.ones_like(data_cp)
        _ = result.get()
        times.append(time.perf_counter() - start)
    mp_time = _median(times)

    times = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        _ = np.ones_like(data_np)
        times.append(time.perf_counter() - start)
    np_time = _median(times)

    return mp_time, np_time


# ===================================================================
# Math benchmarks
# ===================================================================

class _UnaryBench:
    """Picklable callable for unary math benchmarks."""

    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_func = getattr(cp, self.cp_name)
        np_func = getattr(np, self.np_name)

        data_np = np.random.rand(size).astype(np.float32) + 0.1  # avoid log(0)
        data_cp = cp.array(data_np)

        for _ in range(_WARMUP):
            cp_func(data_cp).get()

        times = []
        for _ in range(_REPEATS):
            start = time.perf_counter()
            result = cp_func(data_cp)
            _ = result.get()
            times.append(time.perf_counter() - start)
        mp_time = _median(times)

        times = []
        for _ in range(_REPEATS):
            start = time.perf_counter()
            _ = np_func(data_np)
            times.append(time.perf_counter() - start)
        np_time = _median(times)

        return mp_time, np_time


class _BinaryBench:
    """Picklable callable for binary math benchmarks."""

    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_func = getattr(cp, self.cp_name)
        np_func = getattr(np, self.np_name)

        a_np = np.random.rand(size).astype(np.float32) + 0.1
        b_np = np.random.rand(size).astype(np.float32) + 0.1
        a_cp = cp.array(a_np)
        b_cp = cp.array(b_np)

        for _ in range(_WARMUP):
            cp_func(a_cp, b_cp).get()

        times = []
        for _ in range(_REPEATS):
            start = time.perf_counter()
            result = cp_func(a_cp, b_cp)
            _ = result.get()
            times.append(time.perf_counter() - start)
        mp_time = _median(times)

        times = []
        for _ in range(_REPEATS):
            start = time.perf_counter()
            _ = np_func(a_np, b_np)
            times.append(time.perf_counter() - start)
        np_time = _median(times)

        return mp_time, np_time


bench_sqrt = _UnaryBench("sqrt", "sqrt")
bench_exp = _UnaryBench("exp", "exp")
bench_log = _UnaryBench("log", "log")
bench_sin = _UnaryBench("sin", "sin")
bench_cos = _UnaryBench("cos", "cos")
bench_tan = _UnaryBench("tan", "tan")
bench_abs = _UnaryBench("abs", "abs")

bench_power = _BinaryBench("power", "power")
bench_add = _BinaryBench("add", "add")
bench_subtract = _BinaryBench("subtract", "subtract")
bench_multiply = _BinaryBench("multiply", "multiply")
bench_divide = _BinaryBench("divide", "divide")
bench_floor_divide = _BinaryBench("floor_divide", "floor_divide")
bench_mod = _BinaryBench("mod", "mod")


# ===================================================================
# Reduction benchmarks
# ===================================================================

class _ReductionBench:
    """Picklable callable for reduction benchmarks."""

    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_func = getattr(cp, self.cp_name)
        np_func = getattr(np, self.np_name)

        data_np = np.random.rand(size).astype(np.float32)
        data_cp = cp.array(data_np)

        for _ in range(_WARMUP):
            r = cp_func(data_cp)
            if hasattr(r, 'get'):
                r.get()

        times = []
        for _ in range(_REPEATS):
            start = time.perf_counter()
            r = cp_func(data_cp)
            if hasattr(r, 'get'):
                r.get()
            times.append(time.perf_counter() - start)
        mp_time = _median(times)

        times = []
        for _ in range(_REPEATS):
            start = time.perf_counter()
            _ = np_func(data_np)
            times.append(time.perf_counter() - start)
        np_time = _median(times)

        return mp_time, np_time


class _BoolReductionBench:
    """Picklable callable for boolean reduction benchmarks (any/all)."""

    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_func = getattr(cp, self.cp_name)
        np_func = getattr(np, self.np_name)

        data_np = (np.random.rand(size) > 0.5).astype(np.float32)
        data_cp = cp.array(data_np)

        for _ in range(_WARMUP):
            r = cp_func(data_cp)
            if hasattr(r, 'get'):
                r.get()

        times = []
        for _ in range(_REPEATS):
            start = time.perf_counter()
            r = cp_func(data_cp)
            if hasattr(r, 'get'):
                r.get()
            times.append(time.perf_counter() - start)
        mp_time = _median(times)

        times = []
        for _ in range(_REPEATS):
            start = time.perf_counter()
            _ = np_func(data_np)
            times.append(time.perf_counter() - start)
        np_time = _median(times)

        return mp_time, np_time


bench_sum = _ReductionBench("sum", "sum")
bench_mean = _ReductionBench("mean", "mean")
bench_max = _ReductionBench("max", "max")
bench_min = _ReductionBench("min", "min")
bench_argmax = _ReductionBench("argmax", "argmax")
bench_argmin = _ReductionBench("argmin", "argmin")
bench_std = _ReductionBench("std", "std")
bench_var = _ReductionBench("var", "var")
bench_prod = _ReductionBench("prod", "prod")
bench_cumsum = _ReductionBench("cumsum", "cumsum")
bench_cumprod = _ReductionBench("cumprod", "cumprod")
bench_any = _BoolReductionBench("any", "any")
bench_all = _BoolReductionBench("all", "all")


# ===================================================================
# Comparison benchmarks
# ===================================================================

class _ComparisonBench:
    """Picklable callable for element-wise comparison benchmarks."""

    def __init__(self, cp_name, np_name):
        self.cp_name = cp_name
        self.np_name = np_name

    def __call__(self, size):
        cp_func = getattr(cp, self.cp_name)
        np_func = getattr(np, self.np_name)

        a_np = np.random.rand(size).astype(np.float32)
        b_np = np.random.rand(size).astype(np.float32)
        a_cp = cp.array(a_np)
        b_cp = cp.array(b_np)

        for _ in range(_WARMUP):
            cp_func(a_cp, b_cp).get()

        times = []
        for _ in range(_REPEATS):
            start = time.perf_counter()
            result = cp_func(a_cp, b_cp)
            _ = result.get()
            times.append(time.perf_counter() - start)
        mp_time = _median(times)

        times = []
        for _ in range(_REPEATS):
            start = time.perf_counter()
            _ = np_func(a_np, b_np)
            times.append(time.perf_counter() - start)
        np_time = _median(times)

        return mp_time, np_time


bench_equal = _ComparisonBench("equal", "equal")
bench_not_equal = _ComparisonBench("not_equal", "not_equal")
bench_less = _ComparisonBench("less", "less")
bench_greater = _ComparisonBench("greater", "greater")


# ===================================================================
# BENCHMARKS registry
# ===================================================================

BENCHMARKS = [
    # --- Creation (10) ---
    {"name": "array",      "category": "creation", "func": bench_array,      "sizes": ALL_SIZES},
    {"name": "zeros",      "category": "creation", "func": bench_zeros,      "sizes": ALL_SIZES},
    {"name": "ones",       "category": "creation", "func": bench_ones,       "sizes": ALL_SIZES},
    {"name": "empty",      "category": "creation", "func": bench_empty,      "sizes": ALL_SIZES},
    {"name": "arange",     "category": "creation", "func": bench_arange,     "sizes": ALL_SIZES},
    {"name": "linspace",   "category": "creation", "func": bench_linspace,   "sizes": ALL_SIZES},
    {"name": "eye",        "category": "creation", "func": bench_eye,        "sizes": ALL_SIZES},
    {"name": "full",       "category": "creation", "func": bench_full,       "sizes": ALL_SIZES},
    {"name": "zeros_like", "category": "creation", "func": bench_zeros_like, "sizes": ALL_SIZES},
    {"name": "ones_like",  "category": "creation", "func": bench_ones_like,  "sizes": ALL_SIZES},
    # --- Math (14) ---
    {"name": "sqrt",          "category": "math", "func": bench_sqrt,          "sizes": ALL_SIZES},
    {"name": "exp",           "category": "math", "func": bench_exp,           "sizes": ALL_SIZES},
    {"name": "log",           "category": "math", "func": bench_log,           "sizes": ALL_SIZES},
    {"name": "sin",           "category": "math", "func": bench_sin,           "sizes": ALL_SIZES},
    {"name": "cos",           "category": "math", "func": bench_cos,           "sizes": ALL_SIZES},
    {"name": "tan",           "category": "math", "func": bench_tan,           "sizes": ALL_SIZES},
    {"name": "abs",           "category": "math", "func": bench_abs,           "sizes": ALL_SIZES},
    {"name": "power",         "category": "math", "func": bench_power,         "sizes": ALL_SIZES},
    {"name": "add",           "category": "math", "func": bench_add,           "sizes": ALL_SIZES},
    {"name": "subtract",      "category": "math", "func": bench_subtract,      "sizes": ALL_SIZES},
    {"name": "multiply",      "category": "math", "func": bench_multiply,      "sizes": ALL_SIZES},
    {"name": "divide",        "category": "math", "func": bench_divide,        "sizes": ALL_SIZES},
    {"name": "floor_divide",  "category": "math", "func": bench_floor_divide,  "sizes": ALL_SIZES},
    {"name": "mod",           "category": "math", "func": bench_mod,           "sizes": ALL_SIZES},
    # --- Reductions (13) ---
    {"name": "sum",     "category": "reduction", "func": bench_sum,     "sizes": ALL_SIZES},
    {"name": "mean",    "category": "reduction", "func": bench_mean,    "sizes": ALL_SIZES},
    {"name": "max",     "category": "reduction", "func": bench_max,     "sizes": ALL_SIZES},
    {"name": "min",     "category": "reduction", "func": bench_min,     "sizes": ALL_SIZES},
    {"name": "argmax",  "category": "reduction", "func": bench_argmax,  "sizes": ALL_SIZES},
    {"name": "argmin",  "category": "reduction", "func": bench_argmin,  "sizes": ALL_SIZES},
    {"name": "std",     "category": "reduction", "func": bench_std,     "sizes": ALL_SIZES},
    {"name": "var",     "category": "reduction", "func": bench_var,     "sizes": ALL_SIZES},
    {"name": "prod",    "category": "reduction", "func": bench_prod,    "sizes": ALL_SIZES},
    {"name": "cumsum",  "category": "reduction", "func": bench_cumsum,  "sizes": ALL_SIZES},
    {"name": "cumprod", "category": "reduction", "func": bench_cumprod, "sizes": ALL_SIZES},
    {"name": "any",     "category": "reduction", "func": bench_any,     "sizes": ALL_SIZES},
    {"name": "all",     "category": "reduction", "func": bench_all,     "sizes": ALL_SIZES},
    # --- Comparison (4) ---
    {"name": "equal",     "category": "comparison", "func": bench_equal,     "sizes": ALL_SIZES},
    {"name": "not_equal", "category": "comparison", "func": bench_not_equal, "sizes": ALL_SIZES},
    {"name": "less",      "category": "comparison", "func": bench_less,      "sizes": ALL_SIZES},
    {"name": "greater",   "category": "comparison", "func": bench_greater,   "sizes": ALL_SIZES},
]
