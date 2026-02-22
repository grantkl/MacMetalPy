#!/usr/bin/env python3
"""Comprehensive coverage audit of macmetalpy benchmark file.

Compares ALL public callable names from macmetalpy (top-level + submodules)
against what is actually referenced in benchmarks/bench_vs_numpy.py.
"""

import ast
import sys
import os
import types

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import macmetalpy as mp
import numpy as np
import inspect

# ═══════════════════════════════════════════════════════════════
# 1. Get ALL public callable names from macmetalpy
# ═══════════════════════════════════════════════════════════════

# Top-level
top_level = set()
top_level_all = {}  # name -> object for categorization
for n in dir(mp):
    if n.startswith('_'):
        continue
    obj = getattr(mp, n)
    if callable(obj):
        top_level.add(n)
        top_level_all[n] = obj

# Submodules
linalg_funcs = set()
for n in dir(mp.linalg):
    if not n.startswith('_') and callable(getattr(mp.linalg, n)):
        linalg_funcs.add(f"linalg.{n}")

fft_funcs = set()
for n in dir(mp.fft):
    if not n.startswith('_') and callable(getattr(mp.fft, n)):
        fft_funcs.add(f"fft.{n}")

random_funcs = set()
for n in dir(mp.random):
    if not n.startswith('_') and callable(getattr(mp.random, n)):
        random_funcs.add(f"random.{n}")

all_public = top_level | linalg_funcs | fft_funcs | random_funcs

print("=" * 78)
print("MACMETALPY BENCHMARK COVERAGE AUDIT")
print("=" * 78)
print()
print(f"Public API counts:")
print(f"  Top-level callables:   {len(top_level)}")
print(f"  linalg callables:      {len(linalg_funcs)}")
print(f"  fft callables:         {len(fft_funcs)}")
print(f"  random callables:      {len(random_funcs)}")
print(f"  TOTAL:                 {len(all_public)}")
print()

# ═══════════════════════════════════════════════════════════════
# 2. Parse bench_vs_numpy.py to find ALL mp.X references
# ═══════════════════════════════════════════════════════════════

bench_path = os.path.join(os.path.dirname(__file__), "benchmarks", "bench_vs_numpy.py")
with open(bench_path, "r") as f:
    source = f.read()

tree = ast.parse(source)

benchmarked = set()
method_calls = set()

class MpAttrVisitor(ast.NodeVisitor):
    """Walk AST to find all mp.X and mp.submod.X attribute references."""

    def visit_Attribute(self, node):
        # Check for mp.X (top-level)
        if isinstance(node.value, ast.Name) and node.value.id == 'mp':
            benchmarked.add(node.attr)

        # Check for mp.linalg.X, mp.fft.X, mp.random.X
        if isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.value.id == 'mp':
                submod = node.value.attr
                if submod in ('linalg', 'fft', 'random'):
                    benchmarked.add(f"{submod}.{node.attr}")

        # Check for method calls on arrays
        if isinstance(node.value, ast.Name):
            name = node.value.id
            if any(name.startswith(p) for p in ['cp_', 'mp_']):
                method_calls.add(node.attr)

        self.generic_visit(node)

visitor = MpAttrVisitor()
visitor.visit(tree)

# Also search for string references via regex
import re
method_pattern = re.compile(r'lambda\s+\w+.*?:\s*.*?\.(\w+)\(')
for match in method_pattern.finditer(source):
    method_calls.add(match.group(1))

attr_pattern = re.compile(r'\b(?:cp_\w+|mp_\w+)\.([\w]+)')
for match in attr_pattern.finditer(source):
    method_calls.add(match.group(1))

print("-" * 78)
print("BENCHMARKED mp.X references found in bench_vs_numpy.py:")
print("-" * 78)

benchmarked_top = sorted(b for b in benchmarked if '.' not in b)
benchmarked_linalg = sorted(b for b in benchmarked if b.startswith('linalg.'))
benchmarked_fft = sorted(b for b in benchmarked if b.startswith('fft.'))
benchmarked_random = sorted(b for b in benchmarked if b.startswith('random.'))

print(f"\n  Top-level mp.X ({len(benchmarked_top)}): {', '.join(benchmarked_top)}")
print(f"\n  mp.linalg.X ({len(benchmarked_linalg)}): {', '.join(benchmarked_linalg)}")
print(f"\n  mp.fft.X ({len(benchmarked_fft)}): {', '.join(benchmarked_fft)}")
print(f"\n  mp.random.X ({len(benchmarked_random)}): {', '.join(benchmarked_random)}")
print(f"\n  Array methods: {', '.join(sorted(method_calls))}")

# ═══════════════════════════════════════════════════════════════
# 3. Compute NOT covered and categorize
# ═══════════════════════════════════════════════════════════════

benchmarked_in_api = benchmarked & all_public
not_covered = all_public - benchmarked

# --- Categorization dictionaries ---

# Aliases: function is an alias of another that IS benchmarked
ALIASES = {
    'absolute': 'abs',
    'round_': 'around',
    'round': 'around',
    'amax': 'max',
    'amin': 'min',
    'bitwise_not': 'invert',
    'bitwise_invert': 'invert',
    'bitwise_left_shift': 'left_shift',
    'bitwise_right_shift': 'right_shift',
    'concat': 'concatenate',
    'degrees': 'deg2rad',
    'radians': 'rad2deg',
    'msort': 'sort',
    'random.random_sample': 'random.random',
    'random.ranf': 'random.random',
    'random.sample': 'random.random',
}

# Benchmarked via operators (a + b, a * b, etc.) rather than mp.add(a, b)
BENCHMARKED_VIA_OPERATOR = {
    'add': 'via + operator in run_elementwise_binary',
    'subtract': 'explicitly benchmarked as mp.subtract',  # actually IS benchmarked
    'multiply': 'via * operator in run_elementwise_binary',
    'divide': 'via / operator in run_elementwise_binary',
    'power': 'via ** operator in run_elementwise_binary',
    'floor_divide': 'via // operator in run_elementwise_binary',
    'greater': 'via > operator in run_comparisons',
    'equal': 'via == operator in run_comparisons',
    'logical_and': 'via & operator in run_comparisons',
    'logical_or': 'via | operator in run_comparisons',
}

# Classes / types (not functions you'd benchmark)
# We'll identify these programmatically too
CLASSES_TYPES = set()
# Submodule internal classes
CLASSES_TYPES.update({'linalg.LinAlgError', 'linalg.ndarray', 'fft.ndarray'})

# Check top-level callables that are actually types/classes, not functions
for n in not_covered:
    if '.' in n:
        continue
    obj = top_level_all.get(n)
    if obj is None:
        continue
    # It's a class/type, not a function
    if isinstance(obj, type):
        CLASSES_TYPES.add(n)
    # numpy dtype types are callable but are types, not compute functions
    elif obj is np.float16 or obj is np.float32 or obj is np.float64 or \
         obj is np.int16 or obj is np.int32 or obj is np.int64 or \
         obj is np.uint16 or obj is np.uint32 or obj is np.uint64 or \
         obj is np.bool_ or obj is np.complex64:
        CLASSES_TYPES.add(n)

# Also add numpy scalar types explicitly
NUMPY_DTYPE_TYPES = {
    'bool_', 'float16', 'float32', 'float64', 'complex64',
    'int16', 'int32', 'int64', 'uint16', 'uint32', 'uint64',
}
CLASSES_TYPES.update(NUMPY_DTYPE_TYPES)

# Full list of class/type names
CLASSES_TYPES.update({
    'ndarray', 'RawKernel', 'vectorize',
    'dtype', 'broadcast', 'flatiter', 'nditer',
    'int_', 'float_', 'complex_', 'intp', 'uintp',
    'int8', 'uint8', 'byte', 'ubyte', 'short', 'ushort', 'intc', 'uintc', 'uint',
    'longlong', 'ulonglong',
    'single', 'double', 'half', 'longdouble', 'longfloat',
    'csingle', 'cdouble', 'clongdouble', 'clongfloat', 'singlecomplex', 'longcomplex',
    'complex128', 'cfloat',
    'complexfloating', 'floating', 'integer', 'signedinteger', 'unsignedinteger',
    'number', 'generic', 'inexact',
    'random.Generator', 'random.RandomState',
})

# Not benchmarkable: config, synchronization, meta-ops, index objects
NOT_BENCHMARKABLE = {
    'get_config', 'set_config', 'synchronize',
    'c_', 'r_', 's_', 'mgrid', 'ogrid',
    'einsum_path',
    'matmul',  # benchmarked via @ operator in run_matmul
}

# Skippable: functions noted as skipped in bench file with good reason
SKIPPED_WITH_REASON = {
    'packbits': 'Metal does not support uint8 dtype (noted in bench file)',
    'unpackbits': 'Metal does not support uint8 dtype (noted in bench file)',
    'fromstring': 'np.fromstring binary mode removed in NumPy 2.x (noted in bench file)',
}

# Categorize
categorized = {}
for name in sorted(not_covered):
    if name in ALIASES:
        canonical = ALIASES[name]
        if canonical in benchmarked or canonical in benchmarked_in_api:
            categorized[name] = ("Alias", f"Alias of '{canonical}' (benchmarked)")
        else:
            categorized[name] = ("Alias (gap)", f"Alias of '{canonical}' (canonical NOT benchmarked!)")
    elif name in BENCHMARKED_VIA_OPERATOR:
        categorized[name] = ("Benchmarked via operator", BENCHMARKED_VIA_OPERATOR[name])
    elif name in CLASSES_TYPES:
        categorized[name] = ("Class/type", "Not a compute function")
    elif name in NOT_BENCHMARKABLE:
        categorized[name] = ("Not benchmarkable", "Config/meta/index object")
    elif name in SKIPPED_WITH_REASON:
        categorized[name] = ("Skipped (good reason)", SKIPPED_WITH_REASON[name])
    else:
        categorized[name] = ("COULD STILL BENCHMARK", "Not covered anywhere")

print()
print("=" * 78)
print("NOT DIRECTLY REFERENCED AS mp.X IN BENCHMARK (with categories)")
print("=" * 78)

from collections import defaultdict
by_category = defaultdict(list)
for name, (cat, detail) in categorized.items():
    by_category[cat].append((name, detail))

category_order = [
    "Alias",
    "Benchmarked via operator",
    "Class/type",
    "Not benchmarkable",
    "Skipped (good reason)",
    "Alias (gap)",
    "COULD STILL BENCHMARK",
]
for cat in category_order:
    items = by_category.get(cat, [])
    if not items:
        continue
    print(f"\n  [{cat}] ({len(items)}):")
    for name, detail in sorted(items):
        print(f"    {name:40s} -- {detail}")

# ═══════════════════════════════════════════════════════════════
# 4. Final effective coverage calculation
# ═══════════════════════════════════════════════════════════════

alias_covered = {n for n, (c, _) in categorized.items() if c == 'Alias'}
operator_covered = {n for n, (c, _) in categorized.items() if c == 'Benchmarked via operator'}
class_type_excluded = {n for n, (c, _) in categorized.items() if c == 'Class/type'}
not_benchmarkable_excluded = {n for n, (c, _) in categorized.items() if c == 'Not benchmarkable'}
skipped_excluded = {n for n, (c, _) in categorized.items() if c.startswith('Skipped')}
could_still = {n for n, (c, _) in categorized.items() if c == 'COULD STILL BENCHMARK'}

total_api = len(all_public)
directly_benchmarked = len(benchmarked_in_api)
effectively_benchmarked = directly_benchmarked + len(alias_covered) + len(operator_covered)

excluded = len(class_type_excluded) + len(not_benchmarkable_excluded) + len(skipped_excluded)
effective_api = total_api - len(class_type_excluded) - len(not_benchmarkable_excluded) - len(skipped_excluded)

print()
print("=" * 78)
print("COVERAGE SUMMARY")
print("=" * 78)
print()
print(f"  Total public API callables:              {total_api}")
print(f"  Directly benchmarked (mp.X in AST):      {directly_benchmarked}")
print(f"  Covered via alias:                       {len(alias_covered)}")
print(f"  Covered via operator (a+b, a*b...):      {len(operator_covered)}")
print(f"  EFFECTIVELY BENCHMARKED:                 {effectively_benchmarked}")
print()
print(f"  Excluded (classes/types):                {len(class_type_excluded)}")
print(f"  Excluded (not benchmarkable):            {len(not_benchmarkable_excluded)}")
print(f"  Excluded (skipped, good reason):         {len(skipped_excluded)}")
print(f"  Total excluded:                          {excluded}")
print()
print(f"  Effective API (total - excluded):        {effective_api}")
eff_pct = effectively_benchmarked / effective_api * 100
print(f"  EFFECTIVE COVERAGE:                      {effectively_benchmarked}/{effective_api} = {eff_pct:.1f}%")
print()

raw_pct = directly_benchmarked / total_api * 100
print(f"  Raw coverage (direct/total):             {directly_benchmarked}/{total_api} = {raw_pct:.1f}%")

if could_still:
    print()
    print(f"  *** {len(could_still)} functions still missing from benchmarks: ***")
    for n in sorted(could_still):
        print(f"      - {n}")
else:
    print()
    print("  *** ALL benchmarkable functions are effectively covered! ***")

print()
print("=" * 78)

# Sanity check: all not_covered items should be categorized
uncategorized = not_covered - set(categorized.keys())
if uncategorized:
    print(f"\nWARNING: {len(uncategorized)} uncategorized items: {sorted(uncategorized)}")

# Cross-check: total should add up
accounted = directly_benchmarked + len(alias_covered) + len(operator_covered) + \
            len(class_type_excluded) + len(not_benchmarkable_excluded) + \
            len(skipped_excluded) + len(could_still)
print(f"\nIntegrity check: {accounted} accounted = {directly_benchmarked} direct + "
      f"{len(alias_covered)} alias + {len(operator_covered)} operator + "
      f"{len(class_type_excluded)} class + {len(not_benchmarkable_excluded)} unbenchmarkable + "
      f"{len(skipped_excluded)} skipped + {len(could_still)} gaps")
print(f"Total API: {total_api}  |  Accounted: {accounted}  |  Match: {accounted == total_api}")
