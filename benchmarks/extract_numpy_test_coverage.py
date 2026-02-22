#!/usr/bin/env python3
"""
Extract test coverage information from NumPy 2's installed test suite.

Parses all test_*.py files in NumPy's installed package using the ast module,
extracts test classes and test functions, and maps them to the NumPy APIs
they likely test based on naming conventions.
"""

import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NUMPY_ROOT = Path(np.__path__[0])
OUTPUT_FILE = Path(__file__).parent / "numpy2_test_coverage.txt"


def find_test_files(root: Path) -> list[Path]:
    """Find all test_*.py files under the NumPy installation."""
    test_files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        if dp.name == "tests" or "/tests/" in str(dp):
            for fn in sorted(filenames):
                if fn.startswith("test_") and fn.endswith(".py"):
                    test_files.append(dp / fn)
    return sorted(test_files)


def infer_api_from_name(name: str) -> str:
    """
    Infer the NumPy API being tested from a test function/class name.

    Strips the leading 'test_' or 'Test' prefix and converts CamelCase or
    snake_case remainder to a likely API name.
    """
    # Remove common test prefixes
    if name.startswith("test_"):
        remainder = name[5:]
    elif name.startswith("Test"):
        remainder = name[4:]
    else:
        return name

    if not remainder:
        return name

    # Convert CamelCase to snake_case for class names
    # e.g. TestLinalgSolve -> linalg_solve -> np.linalg.solve
    api = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", remainder).lower()

    # Some common mappings / cleanups
    api = api.strip("_")
    if not api:
        return name

    return api


def extract_tests_from_file(filepath: Path) -> list[dict]:
    """
    Parse a test file with ast and extract test classes and test functions.

    Returns a list of entries:
      { 'class': str|None, 'function': str, 'api': str }
    """
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    entries = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_name = node.name
            class_api = infer_api_from_name(class_name)
            has_test_methods = False
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("test_"):
                        has_test_methods = True
                        func_api = infer_api_from_name(item.name)
                        entries.append({
                            "class": class_name,
                            "function": item.name,
                            "api": func_api,
                            "class_api": class_api,
                        })
            # If the class has no test methods, still record it
            if not has_test_methods:
                entries.append({
                    "class": class_name,
                    "function": None,
                    "api": class_api,
                    "class_api": class_api,
                })

        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # Top-level test function (not inside a class at module level)
            # ast.walk visits all nodes, so we check col_offset for top-level
            if node.col_offset == 0:
                func_api = infer_api_from_name(node.name)
                entries.append({
                    "class": None,
                    "function": node.name,
                    "api": func_api,
                    "class_api": None,
                })

    return entries


def relative_path(filepath: Path) -> str:
    """Return path relative to numpy root's parent (site-packages)."""
    try:
        return str(filepath.relative_to(NUMPY_ROOT.parent))
    except ValueError:
        return str(filepath)


def main():
    test_files = find_test_files(NUMPY_ROOT)
    print(f"Found {len(test_files)} test files in NumPy {np.__version__}")

    # Collect all data
    all_entries = []  # (relative_path, entries)
    api_coverage = defaultdict(list)  # api_name -> [(file, class, func)]
    total_classes = set()
    total_functions = 0

    for tf in test_files:
        entries = extract_tests_from_file(tf)
        rp = relative_path(tf)
        all_entries.append((rp, entries))

        for e in entries:
            if e["class"]:
                total_classes.add((rp, e["class"]))
            if e["function"]:
                total_functions += 1
                api_coverage[e["api"]].append((rp, e["class"], e["function"]))

    # Determine module groupings
    module_stats = defaultdict(lambda: {"files": 0, "classes": 0, "functions": 0, "apis": set()})
    for rp, entries in all_entries:
        # Extract module from path: numpy/linalg/tests/... -> numpy.linalg
        parts = rp.replace("/", ".").split(".tests.")[0]
        # Normalize _core -> core for readability
        display_module = parts.replace("numpy._core", "numpy.core")
        module_stats[display_module]["files"] += 1
        for e in entries:
            if e["class"]:
                module_stats[display_module]["classes"] += 1
            if e["function"]:
                module_stats[display_module]["functions"] += 1
                module_stats[display_module]["apis"].add(e["api"])

    # Write output
    lines = []
    lines.append(f"NumPy {np.__version__} Test Suite Coverage Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total test files:     {len(test_files)}")
    lines.append(f"Total test classes:   {len(total_classes)}")
    lines.append(f"Total test functions: {total_functions}")
    lines.append(f"Unique APIs covered:  {len(api_coverage)}")
    lines.append("")

    # Module summary
    lines.append("=" * 60)
    lines.append("MODULE SUMMARY")
    lines.append("=" * 60)
    for mod in sorted(module_stats):
        s = module_stats[mod]
        lines.append(
            f"  {mod:40s}  files={s['files']:3d}  "
            f"classes={s['classes']:4d}  funcs={s['functions']:5d}  "
            f"apis={len(s['apis']):4d}"
        )
    lines.append("")

    # Top-covered APIs
    lines.append("=" * 60)
    lines.append("TOP 100 MOST-TESTED APIs (by test function count)")
    lines.append("=" * 60)
    sorted_apis = sorted(api_coverage.items(), key=lambda x: -len(x[1]))
    for api_name, refs in sorted_apis[:100]:
        lines.append(f"  {api_name:50s} ({len(refs)} tests)")
    lines.append("")

    # Detailed per-file listing
    lines.append("=" * 60)
    lines.append("DETAILED PER-FILE TEST LISTING")
    lines.append("=" * 60)

    for rp, entries in all_entries:
        if not entries:
            continue
        lines.append("")
        lines.append(f"=== {rp} ===")

        # Group by class
        current_class = None
        for e in entries:
            if e["class"] != current_class:
                current_class = e["class"]
                if current_class:
                    lines.append(f"  {current_class}")
            if e["function"]:
                prefix = "    " if current_class else "  "
                lines.append(f"{prefix}{e['function']} -> [{e['api']}]")

    # Write
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {len(lines)} lines to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
