#!/usr/bin/env python3
"""Compare NumPy 2 APIs with macmetalpy APIs and cross-reference test coverage."""

import re
from pathlib import Path

BASE = Path(__file__).parent

# --- dtype aliases / type names to exclude from "missing" lists ---
DTYPE_ALIASES = {
    'bool', 'bool_', 'byte', 'bytes_', 'cdouble', 'cfloat', 'character',
    'clongdouble', 'clongfloat', 'complex128', 'complex64', 'complex_',
    'complexfloating', 'csingle', 'datetime64', 'double', 'dtype',
    'float16', 'float32', 'float64', 'float_', 'floating', 'generic',
    'half', 'inexact', 'int16', 'int32', 'int64', 'int8', 'int_',
    'intc', 'integer', 'intp', 'long', 'longcomplex', 'longdouble',
    'longfloat', 'longlong', 'number', 'object_', 'short',
    'signedinteger', 'single', 'singlecomplex', 'str_', 'timedelta64',
    'ubyte', 'uint', 'uint16', 'uint32', 'uint64', 'uint8', 'uintc',
    'uintp', 'ulong', 'ulonglong', 'unsignedinteger', 'ushort', 'void',
    # numpy scalar / abstract type hierarchy
    'flexible', 'record',
}

# Modules we skip entirely (testing utilities, string ops, masked arrays, etc.)
SKIP_NUMPY_MODULES = {'numpy.testing', 'numpy.ma', 'numpy.char',
                       'numpy.polynomial', 'numpy.lib', 'numpy.exceptions'}


def parse_numpy_apis(path):
    """Parse numpy2_apis.txt and return {module: {functions: set, classes: set, constants: set}}."""
    apis = {}
    current_module = None
    current_section = None

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            # Module header
            m = re.match(r'^MODULE:\s+(.+)$', line)
            if m:
                current_module = m.group(1).strip()
                apis[current_module] = {'functions': set(), 'classes': set(), 'constants': set()}
                current_section = None
                continue

            if current_module is None:
                continue

            # Section header
            if re.match(r'^\s+Functions\s+\(\d+\):', line):
                current_section = 'functions'
                continue
            elif re.match(r'^\s+Classes\s+\(\d+\):', line):
                current_section = 'classes'
                continue
            elif re.match(r'^\s+Constants\s+\(\d+\):', line):
                current_section = 'constants'
                continue
            elif line.startswith('===') or line.startswith('---'):
                continue

            if current_section in ('functions', 'classes'):
                # Lines look like: "    abs(x, /, ...)" or "    ndarray(...)"
                m2 = re.match(r'^\s{4}(\w+)\(', line)
                if m2:
                    name = m2.group(1)
                    if not name.startswith('_'):
                        apis[current_module][current_section].add(name)
            elif current_section == 'constants':
                m2 = re.match(r'^\s{4}(\w+):', line)
                if m2:
                    name = m2.group(1)
                    if not name.startswith('_'):
                        apis[current_module]['constants'].add(name)

    return apis


def parse_macmetalpy_apis(path):
    """Parse macmetalpy_apis.txt and return {module: {functions: set, classes: set, constants: set}}."""
    apis = {}
    current_module = None
    current_section = None

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            # Module header
            m = re.match(r'^Module:\s+(\S+)', line)
            if m:
                current_module = m.group(1).strip()
                apis[current_module] = {'functions': set(), 'classes': set(), 'constants': set()}
                current_section = None
                continue

            if current_module is None:
                continue

            # Section header
            if re.match(r'^\s+Functions\s+\(\d+\):', line):
                current_section = 'functions'
                continue
            elif re.match(r'^\s+Classes\s+\(\d+\):', line):
                current_section = 'classes'
                continue
            elif re.match(r'^\s+Constants\s+\(\d+\):', line):
                current_section = 'constants'
                continue
            elif re.match(r'^\s+Submodules\s+\(\d+\):', line):
                current_section = 'submodules'
                continue
            elif line.startswith('===') or line.startswith('---'):
                continue

            if current_section in ('functions', 'classes'):
                m2 = re.match(r'^\s{4}(\w+)\(', line)
                if m2:
                    name = m2.group(1)
                    if not name.startswith('_'):
                        apis[current_module][current_section].add(name)
            elif current_section == 'constants':
                m2 = re.match(r'^\s{4}(\w+)\s', line)
                if m2:
                    name = m2.group(1)
                    if not name.startswith('_'):
                        apis[current_module]['constants'].add(name)

    return apis


def parse_test_coverage(path):
    """Parse test_coverage.txt and return a set of API names that appear in test tags."""
    tested = set()
    with open(path) as f:
        for line in f:
            # Match bracketed tags like [np.abs], [np.linalg.svd], etc.
            for m in re.finditer(r'\[np\.(?:[\w.]+\.)?(\w+)\]', line):
                tested.add(m.group(1))
            # Also capture tags like [ndarray.xxx] as ndarray being tested
            for m in re.finditer(r'\[ndarray\.(\w+)', line):
                tested.add(m.group(1))
            # Broader: any [np.xxx] pattern
            for m in re.finditer(r'\[np\.(\w+)\]', line):
                tested.add(m.group(1))
    return tested


def module_mapping():
    """Return list of (display_name, numpy_module, macmetalpy_module) tuples."""
    return [
        ('Top-level (numpy / macmetalpy)', 'numpy', 'macmetalpy'),
        ('linalg', 'numpy.linalg', 'macmetalpy.linalg'),
        ('fft', 'numpy.fft', 'macmetalpy.fft'),
        ('random', 'numpy.random', 'macmetalpy.random'),
    ]


def main():
    np_apis = parse_numpy_apis(BASE / 'numpy2_apis.txt')
    mmp_apis = parse_macmetalpy_apis(BASE / 'macmetalpy_apis.txt')
    tested_names = parse_test_coverage(BASE / 'test_coverage.txt')

    lines = []

    # ==================================================================
    # SECTION 1: Missing from macmetalpy
    # ==================================================================
    lines.append('=' * 72)
    lines.append('MISSING FROM MACMETALPY (in NumPy 2 but not in macmetalpy)')
    lines.append('=' * 72)
    lines.append('')
    lines.append('Only callable functions are listed (dtype aliases, abstract types,')
    lines.append('and private APIs are excluded). Modules numpy.testing, numpy.ma,')
    lines.append('numpy.char, numpy.polynomial, numpy.lib, numpy.exceptions are')
    lines.append('excluded as out-of-scope for a GPU compute library.')
    lines.append('')

    total_missing = 0

    for display, np_mod, mmp_mod in module_mapping():
        np_data = np_apis.get(np_mod, {'functions': set(), 'classes': set(), 'constants': set()})
        mmp_data = mmp_apis.get(mmp_mod, {'functions': set(), 'classes': set(), 'constants': set()})

        # All names in macmetalpy for this module (functions + classes + constants)
        mmp_all = mmp_data['functions'] | mmp_data['classes'] | mmp_data['constants']

        # Functions missing from macmetalpy (only functions, not dtype classes)
        np_funcs = np_data['functions']
        missing_funcs = sorted(np_funcs - mmp_all - DTYPE_ALIASES)

        # Classes missing (exclude dtype aliases)
        np_classes = np_data['classes']
        missing_classes = sorted((np_classes - mmp_all - DTYPE_ALIASES))

        # Constants missing
        np_consts = np_data['constants']
        missing_consts = sorted(np_consts - mmp_all - DTYPE_ALIASES)

        if missing_funcs or missing_classes or missing_consts:
            lines.append(f'--- {display} ---')
            if missing_funcs:
                lines.append(f'  Missing functions ({len(missing_funcs)}):')
                for name in missing_funcs:
                    lines.append(f'    {name}')
            if missing_classes:
                lines.append(f'  Missing classes ({len(missing_classes)}):')
                for name in missing_classes:
                    lines.append(f'    {name}')
            if missing_consts:
                lines.append(f'  Missing constants ({len(missing_consts)}):')
                for name in missing_consts:
                    lines.append(f'    {name}')
            count = len(missing_funcs) + len(missing_classes) + len(missing_consts)
            total_missing += count
            lines.append(f'  Subtotal: {count}')
            lines.append('')

    lines.append(f'TOTAL MISSING: {total_missing}')
    lines.append('')

    # ==================================================================
    # SECTION 2: Extra in macmetalpy
    # ==================================================================
    lines.append('=' * 72)
    lines.append('EXTRA IN MACMETALPY (not in NumPy 2)')
    lines.append('=' * 72)
    lines.append('')
    lines.append('These are APIs exported by macmetalpy that do not exist in NumPy 2.')
    lines.append('They may be macmetalpy-specific extensions or deprecated NumPy names.')
    lines.append('')

    total_extra = 0

    for display, np_mod, mmp_mod in module_mapping():
        np_data = np_apis.get(np_mod, {'functions': set(), 'classes': set(), 'constants': set()})
        mmp_data = mmp_apis.get(mmp_mod, {'functions': set(), 'classes': set(), 'constants': set()})

        np_all = np_data['functions'] | np_data['classes'] | np_data['constants']
        mmp_funcs = mmp_data['functions']
        mmp_classes = mmp_data['classes']

        extra_funcs = sorted((mmp_funcs - np_all - DTYPE_ALIASES))
        extra_classes = sorted((mmp_classes - np_all - DTYPE_ALIASES))

        if extra_funcs or extra_classes:
            lines.append(f'--- {display} ---')
            if extra_funcs:
                lines.append(f'  Extra functions ({len(extra_funcs)}):')
                for name in extra_funcs:
                    lines.append(f'    {name}')
            if extra_classes:
                lines.append(f'  Extra classes ({len(extra_classes)}):')
                for name in extra_classes:
                    lines.append(f'    {name}')
            count = len(extra_funcs) + len(extra_classes)
            total_extra += count
            lines.append(f'  Subtotal: {count}')
            lines.append('')

    lines.append(f'TOTAL EXTRA: {total_extra}')
    lines.append('')

    # ==================================================================
    # SECTION 3: macmetalpy APIs without tests
    # ==================================================================
    lines.append('=' * 72)
    lines.append('MACMETALPY APIs WITHOUT TESTS')
    lines.append('=' * 72)
    lines.append('')
    lines.append('These are functions/classes in macmetalpy that have no corresponding')
    lines.append('test tag [np.<name>] in the test coverage file.')
    lines.append('')

    total_untested = 0

    for display, np_mod, mmp_mod in module_mapping():
        mmp_data = mmp_apis.get(mmp_mod, {'functions': set(), 'classes': set(), 'constants': set()})
        mmp_funcs = mmp_data['functions']

        # Exclude dtype aliases and private names
        relevant = sorted(f for f in mmp_funcs if f not in DTYPE_ALIASES and not f.startswith('_'))
        untested = [f for f in relevant if f not in tested_names]

        if untested:
            lines.append(f'--- {display} ({len(untested)} untested) ---')
            for name in untested:
                lines.append(f'    {name}')
            total_untested += len(untested)
            lines.append('')

    lines.append(f'TOTAL UNTESTED: {total_untested}')
    lines.append('')

    # ==================================================================
    # SECTION 4: Summary
    # ==================================================================
    lines.append('=' * 72)
    lines.append('SUMMARY')
    lines.append('=' * 72)

    for display, np_mod, mmp_mod in module_mapping():
        np_data = np_apis.get(np_mod, {'functions': set(), 'classes': set(), 'constants': set()})
        mmp_data = mmp_apis.get(mmp_mod, {'functions': set(), 'classes': set(), 'constants': set()})
        np_funcs = len(np_data['functions'])
        mmp_funcs = len(mmp_data['functions'])
        pct = (mmp_funcs / np_funcs * 100) if np_funcs > 0 else 0
        lines.append(f'  {display}: {mmp_funcs}/{np_funcs} functions ({pct:.1f}% coverage)')

    lines.append(f'')
    lines.append(f'  APIs missing from macmetalpy: {total_missing}')
    lines.append(f'  Extra macmetalpy APIs:        {total_extra}')
    lines.append(f'  macmetalpy APIs without tests: {total_untested}')
    lines.append('')

    output = '\n'.join(lines)
    out_path = BASE / 'api_comparison.txt'
    out_path.write_text(output)
    print(f'Wrote {out_path} ({len(lines)} lines)')
    print()
    print(output)


if __name__ == '__main__':
    main()
