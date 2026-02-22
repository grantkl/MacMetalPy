#!/usr/bin/env python3
"""
Compare NumPy 2 test suite coverage with macmetalpy test suite coverage.

Parses both test coverage files, cross-references with macmetalpy's API list,
and produces a detailed gap analysis report.

Strategy:
- macmetalpy test_coverage.txt uses qualified tags like [np.zeros], [np.linalg.norm]
- numpy2_test_coverage.txt uses bare descriptive tags like [basic], [copyto]
  So for NumPy, we infer API coverage from:
  1. The test file names (e.g. test_numeric.py -> numeric APIs)
  2. The test function/class names (e.g. test_copyto, TestBroadcastArrays)
  3. Known file-to-API mappings for core NumPy test files
"""

import re
from collections import defaultdict
from pathlib import Path

BENCHMARKS = Path(__file__).parent

NUMPY_COVERAGE = BENCHMARKS / "numpy2_test_coverage.txt"
MACMETALPY_COVERAGE = BENCHMARKS / "test_coverage.txt"
MACMETALPY_APIS = BENCHMARKS / "macmetalpy_apis.txt"
OUTPUT = BENCHMARKS / "test_comparison.txt"


# ---------------------------------------------------------------------------
# 1. Parse macmetalpy API catalog
# ---------------------------------------------------------------------------

def parse_macmetalpy_apis(path: Path) -> dict:
    """Return {module: {kind: [name, ...]}} for each module block."""
    text = path.read_text()
    modules: dict = {}
    current_module = None
    current_kind = None

    for line in text.splitlines():
        m = re.match(r"^Module:\s+(\S+)", line)
        if m:
            current_module = m.group(1)
            modules[current_module] = {"classes": [], "functions": [], "constants": [], "submodules": []}
            current_kind = None
            continue

        if current_module is None:
            continue

        if re.match(r"^\s+Classes\s+\(\d+\):", line):
            current_kind = "classes"
            continue
        if re.match(r"^\s+Functions\s+\(\d+\):", line):
            current_kind = "functions"
            continue
        if re.match(r"^\s+Constants\s+\(\d+\):", line):
            current_kind = "constants"
            continue
        if re.match(r"^\s+Submodules\s+\(\d+\):", line):
            current_kind = "submodules"
            continue

        if current_kind and line.startswith("    "):
            name = line.strip().split("(")[0].split("=")[0].split()[0]
            if name and not name.startswith("-"):
                modules[current_module][current_kind].append(name)

    return modules


def flatten_api_names(modules: dict) -> set:
    """Build a flat set of qualified API names."""
    apis = set()
    for mod, kinds in modules.items():
        for kind in ("functions", "classes", "constants"):
            for name in kinds[kind]:
                if mod == "macmetalpy":
                    apis.add(f"np.{name}")
                elif mod == "macmetalpy.linalg":
                    apis.add(f"np.linalg.{name}")
                elif mod == "macmetalpy.fft":
                    apis.add(f"np.fft.{name}")
                elif mod == "macmetalpy.random":
                    apis.add(f"np.random.{name}")
    return apis


def get_bare_api_names(modules: dict) -> set:
    """Get just the unqualified function/class names."""
    names = set()
    for mod, kinds in modules.items():
        for kind in ("functions", "classes", "constants"):
            for name in kinds[kind]:
                names.add(name)
    return names


# ---------------------------------------------------------------------------
# 2. Parse test coverage files
# ---------------------------------------------------------------------------

def parse_test_coverage(path: Path) -> dict:
    """Parse a test coverage file."""
    text = path.read_text()
    tests = []
    api_tags: dict = defaultdict(int)
    files: set = set()
    classes: set = set()
    current_file = None
    current_class = None

    for line in text.splitlines():
        fm = re.match(r"^=== (.+?) ===", line)
        if fm:
            current_file = fm.group(1)
            files.add(current_file)
            current_class = None
            continue

        cm = re.match(r"^  (\S+)$", line)
        if cm and current_file and not line.strip().startswith("test_"):
            candidate = cm.group(1)
            if candidate.startswith("Test") or candidate == "(module-level)":
                current_class = candidate
                classes.add(candidate)
                continue

        tm = re.match(r"^\s+test_\S+.*->\s+\[(.+?)\]", line)
        if tm:
            tag = tm.group(1)
            test_name_match = re.match(r"^\s+(test_\S+)", line)
            test_name = test_name_match.group(1) if test_name_match else "unknown"
            tests.append((current_file, current_class, test_name, tag))
            api_tags[tag] += 1
            continue

    return {
        "test_functions": tests,
        "api_tags": dict(api_tags),
        "files": files,
        "classes": classes,
        "total_tests": len(tests),
    }


# ---------------------------------------------------------------------------
# 3. Extract tested APIs from macmetalpy coverage (uses np.xxx tags directly)
# ---------------------------------------------------------------------------

def extract_mmp_tested_apis(coverage: dict) -> set:
    """Extract qualified API names from macmetalpy test coverage."""
    apis = set()
    for _file, _cls, _test, tag in coverage["test_functions"]:
        tag = tag.strip()
        # Handle slashed alternatives: "np.reshape / ndarray.reshape"
        parts = [p.strip() for p in tag.split("/")]
        for p in parts:
            if p.startswith("np."):
                apis.add(p)
    return apis


# ---------------------------------------------------------------------------
# 4. Infer NumPy-tested APIs from file names, class names, and test names
# ---------------------------------------------------------------------------

# Map NumPy test file paths to the API areas they cover
NUMPY_FILE_TO_AREA = {
    "test_multiarray.py": "core",
    "test_umath.py": "ufunc",
    "test_numeric.py": "numeric",
    "test_numerictypes.py": "dtypes",
    "test_shape_base.py": "manipulation",
    "test_function_base.py": "creation",
    "test_fromnumeric.py": "core",
    "test_arrayprint.py": "formatting",
    "test_indexing.py": "indexing",
    "test_linalg.py": "linalg",
    "test_fft.py": "fft",
    "test_random.py": "random",
}


def infer_numpy_tested_apis(numpy_cov: dict, catalog_bare_names: set,
                            catalog_qualified: set) -> set:
    """Infer which macmetalpy-catalog APIs are tested by NumPy's test suite.

    Strategy: scan NumPy test function names and class names for references
    to known API names. E.g. if NumPy has test_copyto or TestCopyto, then
    we infer that 'copyto' is tested.
    """
    # Build a lookup: bare_name -> set of qualified names
    bare_to_qualified: dict = defaultdict(set)
    for qname in catalog_qualified:
        # np.linalg.norm -> norm
        bare = qname.split(".")[-1]
        bare_to_qualified[bare].add(qname)

    tested = set()

    for _file, cls_name, test_name, _tag in numpy_cov["test_functions"]:
        # Extract candidate API names from the test function name
        # test_copyto -> copyto
        # test_broadcast_arrays -> broadcast_arrays
        # test_array_astype -> astype (and array)
        fn_body = test_name.replace("test_", "", 1) if test_name.startswith("test_") else test_name

        # Try matching the full body first
        if fn_body in bare_to_qualified:
            tested.update(bare_to_qualified[fn_body])

        # Try each word/segment
        for segment in fn_body.split("_"):
            if segment in bare_to_qualified:
                tested.update(bare_to_qualified[segment])

        # Also try class name: TestCopyto -> copyto
        if cls_name and cls_name.startswith("Test"):
            cls_body = cls_name[4:]  # Remove "Test"
            # CamelCase to snake_case
            snake = re.sub(r'(?<!^)(?=[A-Z])', '_', cls_body).lower()
            if snake in bare_to_qualified:
                tested.update(bare_to_qualified[snake])
            # Also try lowercase direct
            lower = cls_body.lower()
            if lower in bare_to_qualified:
                tested.update(bare_to_qualified[lower])

        # Check file path for submodule hints
        file_path = _file if _file else ""
        if "linalg" in file_path:
            # If a linalg-related test mentions a name, prefer np.linalg.xxx
            pass  # The bare_to_qualified lookup already maps correctly
        if "fft" in file_path:
            pass
        if "random" in file_path:
            pass

    return tested


# ---------------------------------------------------------------------------
# 5. Build the comparison report
# ---------------------------------------------------------------------------

def build_report(
    mmp_modules: dict,
    catalog_qualified: set,
    numpy_cov: dict,
    mmp_cov: dict,
    catalog_bare_names: set,
) -> str:
    lines = []

    # Extract API sets
    mmp_tested = extract_mmp_tested_apis(mmp_cov)
    numpy_tested = infer_numpy_tested_apis(numpy_cov, catalog_bare_names, catalog_qualified)

    lines.append("=" * 72)
    lines.append("Test Coverage Comparison: NumPy 2 vs macmetalpy")
    lines.append("=" * 72)
    lines.append("")

    # --- Summary stats ---
    lines.append("-" * 72)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 72)
    lines.append(f"  NumPy 2 test suite:")
    lines.append(f"    Total test files:      {len(numpy_cov['files'])}")
    lines.append(f"    Total test classes:     {len(numpy_cov['classes'])}")
    lines.append(f"    Total test functions:   {numpy_cov['total_tests']}")
    lines.append(f"    Unique API tags:        {len(numpy_cov['api_tags'])}")
    lines.append(f"")
    lines.append(f"  macmetalpy test suite:")
    lines.append(f"    Total test files:      {len(mmp_cov['files'])}")
    lines.append(f"    Total test classes:     {len(mmp_cov['classes'])}")
    lines.append(f"    Total test functions:   {mmp_cov['total_tests']}")
    lines.append(f"    Unique API tags:        {len(mmp_cov['api_tags'])}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("API-LEVEL TEST COVERAGE (against macmetalpy catalog)")
    lines.append("-" * 72)
    lines.append(f"  macmetalpy catalog size:              {len(catalog_qualified)}")
    lines.append(f"  Catalog APIs tested by macmetalpy:    {len(mmp_tested & catalog_qualified)}")
    lines.append(f"  Catalog APIs tested by NumPy 2:       {len(numpy_tested & catalog_qualified)}")
    lines.append(f"  Catalog APIs tested by both:          {len(mmp_tested & numpy_tested & catalog_qualified)}")
    lines.append(f"  Catalog APIs tested by neither:       {len(catalog_qualified - mmp_tested - numpy_tested)}")

    pct_mmp = 100 * len(mmp_tested & catalog_qualified) / len(catalog_qualified) if catalog_qualified else 0
    pct_np = 100 * len(numpy_tested & catalog_qualified) / len(catalog_qualified) if catalog_qualified else 0
    pct_either = 100 * len((mmp_tested | numpy_tested) & catalog_qualified) / len(catalog_qualified) if catalog_qualified else 0
    lines.append(f"")
    lines.append(f"  macmetalpy coverage:   {pct_mmp:.1f}%")
    lines.append(f"  NumPy 2 coverage:      {pct_np:.1f}%")
    lines.append(f"  Combined coverage:     {pct_either:.1f}%")
    lines.append("")

    # ===================================================================
    # Section A: APIs that NumPy 2 tests but macmetalpy does NOT
    # ===================================================================
    lines.append("=" * 72)
    lines.append("A. COVERAGE GAPS: NumPy 2 tests these catalog APIs but macmetalpy")
    lines.append("   does NOT (highest-priority gaps to close)")
    lines.append("=" * 72)

    gap_a = sorted((numpy_tested & catalog_qualified) - mmp_tested)
    lines.append(f"\n  Total: {len(gap_a)} APIs")
    lines.append("")

    # Group by submodule
    groups_a: dict = defaultdict(list)
    for api in gap_a:
        if api.startswith("np.linalg."):
            groups_a["np.linalg"].append(api)
        elif api.startswith("np.fft."):
            groups_a["np.fft"].append(api)
        elif api.startswith("np.random."):
            groups_a["np.random"].append(api)
        else:
            groups_a["np (top-level)"].append(api)

    for group_name in sorted(groups_a.keys()):
        apis_in_group = groups_a[group_name]
        lines.append(f"  {group_name} ({len(apis_in_group)}):")
        for api in sorted(apis_in_group):
            lines.append(f"    {api}")
        lines.append("")

    # ===================================================================
    # Section B: APIs macmetalpy tests but NumPy 2 does NOT
    # ===================================================================
    lines.append("=" * 72)
    lines.append("B. macmetalpy tests these catalog APIs but NumPy 2 does NOT")
    lines.append("   (unique macmetalpy coverage)")
    lines.append("=" * 72)

    gap_b = sorted((mmp_tested & catalog_qualified) - numpy_tested)
    lines.append(f"\n  Total: {len(gap_b)} APIs")
    lines.append("")

    groups_b: dict = defaultdict(list)
    for api in gap_b:
        if api.startswith("np.linalg."):
            groups_b["np.linalg"].append(api)
        elif api.startswith("np.fft."):
            groups_b["np.fft"].append(api)
        elif api.startswith("np.random."):
            groups_b["np.random"].append(api)
        else:
            groups_b["np (top-level)"].append(api)

    for group_name in sorted(groups_b.keys()):
        apis_in_group = groups_b[group_name]
        lines.append(f"  {group_name} ({len(apis_in_group)}):")
        for api in sorted(apis_in_group):
            # Count how many tests macmetalpy has
            mmp_count = 0
            for tag, cnt in mmp_cov["api_tags"].items():
                parts = [p.strip() for p in tag.split("/")]
                for p in parts:
                    if p == api:
                        mmp_count += cnt
            lines.append(f"    {api:<45s}  ({mmp_count} macmetalpy tests)")
        lines.append("")

    # ===================================================================
    # Section C: Catalog APIs with NO tests at all
    # ===================================================================
    lines.append("=" * 72)
    lines.append("C. UNTESTED: macmetalpy catalog APIs with NO tests in either suite")
    lines.append("=" * 72)

    untested = sorted(catalog_qualified - mmp_tested - numpy_tested)
    lines.append(f"\n  Total: {len(untested)} APIs")
    lines.append("")

    groups_c: dict = defaultdict(list)
    for api in untested:
        if api.startswith("np.linalg."):
            groups_c["np.linalg"].append(api)
        elif api.startswith("np.fft."):
            groups_c["np.fft"].append(api)
        elif api.startswith("np.random."):
            groups_c["np.random"].append(api)
        else:
            groups_c["np (top-level)"].append(api)

    for group_name in sorted(groups_c.keys()):
        apis_in_group = groups_c[group_name]
        lines.append(f"  {group_name} ({len(apis_in_group)}):")
        for api in sorted(apis_in_group):
            lines.append(f"    {api}")
        lines.append("")

    # ===================================================================
    # Section D: Coverage by macmetalpy module
    # ===================================================================
    lines.append("=" * 72)
    lines.append("D. COVERAGE BY macmetalpy MODULE")
    lines.append("=" * 72)
    lines.append("")

    for mod_name, kinds in mmp_modules.items():
        all_funcs = []
        prefix = "np."
        if mod_name == "macmetalpy.linalg":
            prefix = "np.linalg."
        elif mod_name == "macmetalpy.fft":
            prefix = "np.fft."
        elif mod_name == "macmetalpy.random":
            prefix = "np.random."

        for kind in ("functions", "classes"):
            for name in kinds[kind]:
                all_funcs.append(prefix + name)

        if not all_funcs:
            continue

        tested_mmp_list = [f for f in all_funcs if f in mmp_tested]
        tested_numpy_list = [f for f in all_funcs if f in numpy_tested]
        tested_both = [f for f in all_funcs if f in mmp_tested and f in numpy_tested]
        untested_list = [f for f in all_funcs if f not in mmp_tested and f not in numpy_tested]

        pct = 100 * len(tested_mmp_list) / len(all_funcs) if all_funcs else 0
        pct_np = 100 * len(tested_numpy_list) / len(all_funcs) if all_funcs else 0

        lines.append(f"  {mod_name}:")
        lines.append(f"    Total APIs:              {len(all_funcs)}")
        lines.append(f"    Tested by macmetalpy:    {len(tested_mmp_list)} ({pct:.0f}%)")
        lines.append(f"    Tested by NumPy 2:       {len(tested_numpy_list)} ({pct_np:.0f}%)")
        lines.append(f"    Tested by both:          {len(tested_both)}")
        lines.append(f"    Untested by either:      {len(untested_list)}")

        if untested_list:
            lines.append(f"    Untested APIs:")
            for f in sorted(untested_list):
                lines.append(f"      {f}")
        lines.append("")

    # ===================================================================
    # Section E: APIs tested by both (well-covered)
    # ===================================================================
    lines.append("=" * 72)
    lines.append("E. WELL-COVERED: Catalog APIs tested by BOTH suites")
    lines.append("=" * 72)

    both_tested = sorted(mmp_tested & numpy_tested & catalog_qualified)
    lines.append(f"\n  Total: {len(both_tested)} APIs")
    lines.append("")
    for api in both_tested:
        lines.append(f"    {api}")
    lines.append("")

    # ===================================================================
    # Section F: macmetalpy test tag distribution
    # ===================================================================
    lines.append("=" * 72)
    lines.append("F. macmetalpy TEST TAG DISTRIBUTION (top 80)")
    lines.append("=" * 72)
    lines.append("")

    sorted_tags = sorted(mmp_cov["api_tags"].items(), key=lambda x: -x[1])
    lines.append(f"  {'API Tag':<50s}  {'Count':>7s}")
    lines.append(f"  {'-'*50}  {'-'*7}")
    for tag, count in sorted_tags[:80]:
        lines.append(f"  {tag:<50s}  {count:>7d}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Parsing macmetalpy API catalog...")
    mmp_modules = parse_macmetalpy_apis(MACMETALPY_APIS)
    catalog_qualified = flatten_api_names(mmp_modules)
    catalog_bare = get_bare_api_names(mmp_modules)
    print(f"  Found {len(catalog_qualified)} qualified APIs, {len(catalog_bare)} bare names")

    print("Parsing NumPy 2 test coverage...")
    numpy_cov = parse_test_coverage(NUMPY_COVERAGE)
    print(f"  Found {numpy_cov['total_tests']} tests across {len(numpy_cov['files'])} files")

    print("Parsing macmetalpy test coverage...")
    mmp_cov = parse_test_coverage(MACMETALPY_COVERAGE)
    print(f"  Found {mmp_cov['total_tests']} tests across {len(mmp_cov['files'])} files")

    print("Building comparison report...")
    report = build_report(mmp_modules, catalog_qualified, numpy_cov, mmp_cov, catalog_bare)

    OUTPUT.write_text(report)
    print(f"Report written to {OUTPUT}")
    print()

    # Print a quick summary to stdout
    for line in report.split("\n")[:80]:
        print(line)
    print("... (see full report in file)")


if __name__ == "__main__":
    main()
