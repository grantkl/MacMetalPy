#!/usr/bin/env python3
"""Enumerate all public APIs in macmetalpy and its submodules."""

import sys
import types
import inspect

sys.path.insert(0, "src")
import macmetalpy as cp


def classify(module, names):
    """Classify names into functions, classes, and constants/other."""
    functions = []
    classes = []
    constants = []
    for name in sorted(names):
        obj = getattr(module, name)
        if isinstance(obj, type):
            classes.append(name)
        elif callable(obj) or isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
            functions.append(name)
        elif isinstance(obj, types.ModuleType):
            # Skip submodules — they get their own section
            continue
        else:
            constants.append(name)
    return functions, classes, constants


def public_names(module):
    """Return non-private names from a module."""
    return [n for n in dir(module) if not n.startswith("_")]


def format_section(title, module):
    lines = [f"=== {title} ==="]
    names = public_names(module)
    functions, classes, constants = classify(module, names)
    lines.append("Functions:")
    for f in functions:
        lines.append(f"  {f}")
    lines.append("")
    lines.append("Classes:")
    for c in classes:
        lines.append(f"  {c}")
    lines.append("")
    lines.append("Constants:")
    for c in constants:
        lines.append(f"  {c}")
    lines.append("")
    total = len(functions) + len(classes) + len(constants)
    lines.append(f"Total: {total} public APIs ({len(functions)} functions, {len(classes)} classes, {len(constants)} constants)")
    lines.append("")
    return lines


sections = [
    ("macmetalpy (top-level)", cp),
    ("macmetalpy.linalg", cp.linalg),
    ("macmetalpy.fft", cp.fft),
    ("macmetalpy.random", cp.random),
]

all_lines = []
grand_total = 0
for title, module in sections:
    section_lines = format_section(title, module)
    all_lines.extend(section_lines)
    all_lines.append("")

output = "\n".join(all_lines)
print(output)

with open("benchmarks/macmetalpy_apis_v2.txt", "w") as f:
    f.write(output + "\n")

print("\nSaved to benchmarks/macmetalpy_apis_v2.txt")
