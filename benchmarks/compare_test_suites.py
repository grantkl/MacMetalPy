#!/usr/bin/env python3
"""
Compare NumPy 2 test suite coverage with macmetalpy test suite coverage.

Reads:
  - benchmarks/numpy2_test_coverage.txt  (NumPy 2.4.2 test coverage)
  - benchmarks/test_coverage.txt         (macmetalpy test coverage)

Writes:
  - benchmarks/test_comparison.txt       (comparison report)
"""

import re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
NUMPY_FILE = BASE / "numpy2_test_coverage.txt"
MACMETAL_FILE = BASE / "test_coverage.txt"
OUTPUT_FILE = BASE / "test_comparison.txt"


def extract_apis_from_numpy(filepath: Path) -> dict[str, list[str]]:
    """
    Extract API -> test references from the NumPy coverage file.
    Returns {api_name: [(file, class, function), ...]}.
    """
    apis = defaultdict(list)
    current_file = ""
    current_class = ""

    for line in filepath.read_text().splitlines():
        # File header: === numpy/linalg/tests/test_linalg.py ===
        m = re.match(r"^=== (.+) ===$", line)
        if m:
            current_file = m.group(1)
            current_class = ""
            continue

        # Class header (indented by 2 spaces, no ->)
        m = re.match(r"^  (\w+)$", line)
        if m and not "->" in line:
            current_class = m.group(1)
            continue

        # Test function -> [api]
        m = re.match(r"^\s+(test_\w+)\s+->\s+\[(.+)\]$", line)
        if m:
            func_name = m.group(1)
            api_name = m.group(2)
            apis[api_name].append((current_file, current_class, func_name))

    return apis


def extract_apis_from_macmetal(filepath: Path) -> dict[str, list[str]]:
    """
    Extract API -> test references from the macmetalpy coverage file.
    The macmetalpy file has API names like 'np.sum', 'np.linalg.solve', etc.
    Returns {api_name: [(file, class, function), ...]}.
    """
    apis = defaultdict(list)
    current_file = ""
    current_class = ""

    for line in filepath.read_text().splitlines():
        # File header
        m = re.match(r"^=== (.+) ===$", line)
        if m:
            current_file = m.group(1)
            current_class = ""
            continue

        # Class header (indented by 2 spaces, no ->)
        m = re.match(r"^  (\w+)$", line)
        if m and "->" not in line:
            current_class = m.group(1)
            continue

        # Test function -> [api]
        m = re.match(r"^\s+(test_\w+)\s+->\s+\[(.+)\]$", line)
        if m:
            func_name = m.group(1)
            api_name = m.group(2)
            apis[api_name].append((current_file, current_class, func_name))

    return apis


def normalize_numpy_api(api_name: str) -> str:
    """Normalize a NumPy test API name (snake_case from test name) to a canonical form."""
    return api_name.lower().strip()


def normalize_macmetal_api(api_name: str) -> str:
    """Normalize a macmetalpy API reference to a canonical form for comparison."""
    name = api_name.strip()
    # Remove 'np.' prefix
    if name.startswith("np."):
        name = name[3:]
    # Remove 'ndarray.' prefix
    if name.startswith("ndarray."):
        name = name[8:]
    return name.lower()


# Build a mapping of well-known NumPy API categories for richer comparison
NUMPY_API_CATEGORIES = {
    "creation": [
        "array", "zeros", "ones", "empty", "full", "arange", "linspace",
        "logspace", "geomspace", "eye", "identity", "diag", "diagflat",
        "tri", "triu", "tril", "vander", "indices", "fromfunction",
        "fromiter", "frombuffer", "fromstring", "meshgrid", "zeros_like",
        "ones_like", "empty_like", "full_like", "copy", "asarray",
        "asanyarray", "ascontiguousarray",
    ],
    "elementwise_math": [
        "add", "subtract", "multiply", "divide", "true_divide",
        "floor_divide", "negative", "positive", "power", "remainder",
        "mod", "fmod", "absolute", "abs", "sqrt", "square", "cbrt",
        "reciprocal", "exp", "exp2", "expm1", "log", "log2", "log10",
        "log1p", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
        "arctan2", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
        "floor", "ceil", "trunc", "rint", "round", "around", "fix",
        "sign", "heaviside", "maximum", "minimum", "fmax", "fmin",
        "clip",
    ],
    "reduction": [
        "sum", "prod", "mean", "std", "var", "min", "max", "argmin",
        "argmax", "all", "any", "nansum", "nanprod", "nanmean", "nanstd",
        "nanvar", "nanmin", "nanmax", "nanargmin", "nanargmax",
        "cumsum", "cumprod", "nancumsum", "nancumprod",
        "count_nonzero", "median", "nanmedian", "percentile",
        "nanpercentile", "quantile", "nanquantile",
    ],
    "linalg": [
        "dot", "matmul", "inner", "outer", "vdot", "tensordot", "einsum",
        "kron", "cross", "linalg.norm", "linalg.inv", "linalg.det",
        "linalg.solve", "linalg.eig", "linalg.eigh", "linalg.eigvals",
        "linalg.eigvalsh", "linalg.svd", "linalg.cholesky", "linalg.qr",
        "linalg.lstsq", "linalg.pinv", "linalg.matrix_rank",
        "linalg.slogdet", "linalg.cond", "linalg.matrix_power",
        "trace", "diagonal",
    ],
    "fft": [
        "fft.fft", "fft.ifft", "fft.fft2", "fft.ifft2", "fft.fftn",
        "fft.ifftn", "fft.rfft", "fft.irfft", "fft.rfft2", "fft.irfft2",
        "fft.rfftn", "fft.irfftn", "fft.hfft", "fft.ihfft",
        "fft.fftfreq", "fft.rfftfreq", "fft.fftshift", "fft.ifftshift",
    ],
    "shape_manipulation": [
        "reshape", "ravel", "flatten", "transpose", "swapaxes",
        "moveaxis", "squeeze", "expand_dims", "broadcast_to",
        "broadcast_arrays", "atleast_1d", "atleast_2d", "atleast_3d",
    ],
    "indexing": [
        "take", "put", "choose", "compress", "where", "nonzero",
        "argwhere", "flatnonzero", "searchsorted", "extract",
        "place", "select", "piecewise",
    ],
    "sorting": [
        "sort", "argsort", "partition", "argpartition", "lexsort",
        "unique",
    ],
    "joining_splitting": [
        "concatenate", "stack", "vstack", "hstack", "dstack",
        "column_stack", "row_stack", "split", "array_split",
        "hsplit", "vsplit", "dsplit", "tile", "repeat",
    ],
    "comparison_logic": [
        "equal", "not_equal", "less", "less_equal", "greater",
        "greater_equal", "logical_and", "logical_or", "logical_not",
        "logical_xor", "isnan", "isinf", "isfinite", "isnat",
        "allclose", "array_equal", "array_equiv",
    ],
    "statistics": [
        "average", "histogram", "histogram2d", "histogramdd",
        "bincount", "digitize", "corrcoef", "cov",
    ],
    "io": [
        "save", "load", "savez", "savez_compressed", "savetxt",
        "loadtxt", "genfromtxt", "fromfile", "tofile",
    ],
    "random": [
        "random.rand", "random.randn", "random.randint",
        "random.random", "random.uniform", "random.normal",
        "random.seed", "random.shuffle", "random.permutation",
        "random.choice",
    ],
    "dtype_utility": [
        "dtype", "can_cast", "promote_types", "result_type",
        "common_type", "min_scalar_type", "finfo", "iinfo",
        "issubdtype",
    ],
}


def find_macmetal_apis_for_category(macmetal_apis: dict, category_apis: list[str]) -> dict[str, list]:
    """Find which APIs in a category are tested in macmetalpy."""
    found = {}
    for api in category_apis:
        # Try multiple matching strategies
        matches = []
        # Direct match (e.g., "np.sum" -> normalized "sum")
        norm = api.lower()
        for mac_api, refs in macmetal_apis.items():
            mac_norm = normalize_macmetal_api(mac_api)
            if mac_norm == norm or mac_norm == api:
                matches.extend(refs)
        found[api] = matches
    return found


def find_numpy_apis_for_category(numpy_apis: dict, category_apis: list[str]) -> dict[str, list]:
    """Find which APIs in a category are tested in NumPy's test suite."""
    found = {}
    for api in category_apis:
        matches = []
        # The numpy coverage uses bare names like "sum", "fft", "solve"
        # We need to match against the leaf name
        leaf = api.split(".")[-1].lower()
        for np_api, refs in numpy_apis.items():
            np_norm = normalize_numpy_api(np_api)
            if np_norm == leaf or np_norm == api.lower():
                matches.extend(refs)
        found[api] = matches
    return found


def main():
    numpy_apis = extract_apis_from_numpy(NUMPY_FILE)
    macmetal_apis = extract_apis_from_macmetal(MACMETAL_FILE)

    lines = []
    lines.append("Test Suite Comparison: NumPy 2.4.2 vs macmetalpy")
    lines.append("=" * 70)
    lines.append("")

    # Summary stats
    numpy_total_tests = sum(len(v) for v in numpy_apis.values())
    macmetal_total_tests = sum(len(v) for v in macmetal_apis.values())
    lines.append(f"NumPy 2 unique API tags:      {len(numpy_apis)}")
    lines.append(f"NumPy 2 total test references: {numpy_total_tests}")
    lines.append(f"macmetalpy unique API tags:    {len(macmetal_apis)}")
    lines.append(f"macmetalpy total test refs:    {macmetal_total_tests}")
    lines.append("")

    # Category-by-category comparison
    lines.append("=" * 70)
    lines.append("CATEGORY-BY-CATEGORY COMPARISON")
    lines.append("=" * 70)
    lines.append("")

    overall_numpy_covered = 0
    overall_macmetal_covered = 0
    overall_total = 0
    gap_apis = []  # APIs tested in NumPy but not macmetalpy

    for category, api_list in NUMPY_API_CATEGORIES.items():
        numpy_found = find_numpy_apis_for_category(numpy_apis, api_list)
        macmetal_found = find_macmetal_apis_for_category(macmetal_apis, api_list)

        numpy_covered = sum(1 for v in numpy_found.values() if v)
        macmetal_covered = sum(1 for v in macmetal_found.values() if v)
        total = len(api_list)

        overall_numpy_covered += numpy_covered
        overall_macmetal_covered += macmetal_covered
        overall_total += total

        lines.append(f"--- {category.upper().replace('_', ' ')} ({total} APIs) ---")
        lines.append(f"  NumPy 2 covers:    {numpy_covered}/{total}")
        lines.append(f"  macmetalpy covers: {macmetal_covered}/{total}")

        # Find gaps: in NumPy but not macmetalpy
        numpy_only = []
        macmetal_only = []
        both = []
        neither = []

        for api in api_list:
            has_numpy = bool(numpy_found.get(api))
            has_macmetal = bool(macmetal_found.get(api))
            if has_numpy and has_macmetal:
                both.append(api)
            elif has_numpy and not has_macmetal:
                numpy_only.append(api)
                gap_apis.append((category, api))
            elif has_macmetal and not has_numpy:
                macmetal_only.append(api)
            else:
                neither.append(api)

        if both:
            lines.append(f"  Both test:         {', '.join(both)}")
        if numpy_only:
            lines.append(f"  NumPy only:        {', '.join(numpy_only)}")
        if macmetal_only:
            lines.append(f"  macmetalpy only:   {', '.join(macmetal_only)}")
        if neither:
            lines.append(f"  Neither tests:     {', '.join(neither)}")
        lines.append("")

    # Overall summary
    lines.append("=" * 70)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total well-known APIs tracked:  {overall_total}")
    lines.append(f"NumPy 2 test coverage:          {overall_numpy_covered}/{overall_total} ({100*overall_numpy_covered/overall_total:.1f}%)")
    lines.append(f"macmetalpy test coverage:        {overall_macmetal_covered}/{overall_total} ({100*overall_macmetal_covered/overall_total:.1f}%)")
    lines.append(f"Coverage gap (NumPy but not macmetalpy): {len(gap_apis)} APIs")
    lines.append("")

    # Detailed gap listing
    lines.append("=" * 70)
    lines.append("COVERAGE GAPS: APIs tested in NumPy 2 but NOT in macmetalpy")
    lines.append("=" * 70)
    gap_by_category = defaultdict(list)
    for cat, api in gap_apis:
        gap_by_category[cat].append(api)

    for cat in sorted(gap_by_category):
        apis = gap_by_category[cat]
        lines.append(f"  {cat}: {', '.join(apis)}")
    lines.append("")

    # macmetalpy-specific API tags (not standard NumPy test names)
    lines.append("=" * 70)
    lines.append("macmetalpy-SPECIFIC API TAGS (unique to macmetalpy tests)")
    lines.append("=" * 70)
    macmetal_specific_tags = set()
    for api in macmetal_apis:
        norm = normalize_macmetal_api(api)
        if norm in ("unknown", "dtype system", "error handling", "configuration",
                     "broadcasting", "complex number ops", "edge cases",
                     "i/o operations", "operator *", "operator -",
                     "operator /", "operator **", "operator - (unary)"):
            continue
        # Check if this looks like a macmetalpy-specific concept
        if "device" in norm or "metal" in norm or "gpu" in norm or "macmetal" in norm:
            macmetal_specific_tags.add(api)
    for tag in sorted(macmetal_specific_tags):
        refs = macmetal_apis[tag]
        lines.append(f"  {tag} ({len(refs)} tests)")
    if not macmetal_specific_tags:
        lines.append("  (none found with explicit device/metal/gpu tags)")
    lines.append("")

    # Full listing of all macmetalpy API tags and their test counts
    lines.append("=" * 70)
    lines.append("ALL macmetalpy API TAGS (sorted by test count)")
    lines.append("=" * 70)
    sorted_mac = sorted(macmetal_apis.items(), key=lambda x: -len(x[1]))
    for api, refs in sorted_mac:
        lines.append(f"  {api:50s} ({len(refs)} tests)")
    lines.append("")

    # Full listing of all NumPy API tags and their test counts (top 200)
    lines.append("=" * 70)
    lines.append("TOP 200 NumPy 2 API TAGS (sorted by test count)")
    lines.append("=" * 70)
    sorted_np = sorted(numpy_apis.items(), key=lambda x: -len(x[1]))
    for api, refs in sorted_np[:200]:
        lines.append(f"  {api:50s} ({len(refs)} tests)")
    lines.append("")

    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {len(lines)} lines to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
