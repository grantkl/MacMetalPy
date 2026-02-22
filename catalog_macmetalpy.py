#!/usr/bin/env python3
"""Catalog all public APIs in macmetalpy, writing JSON output."""

import importlib
import inspect
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def get_params(obj):
    """Get parameter names for a callable, or None if uninspectable."""
    try:
        sig = inspect.signature(obj)
        return list(sig.parameters.keys())
    except (ValueError, TypeError):
        return None

def catalog_module(mod, prefix=""):
    """Return dict of {name: {type, params}} for all public items."""
    items = {}
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        kind = "unknown"
        if inspect.ismodule(obj):
            kind = "module"
        elif inspect.isclass(obj):
            kind = "class"
        elif callable(obj):
            kind = "function"
        else:
            kind = "constant"

        params = None
        if kind == "function":
            params = get_params(obj)

        items[name] = {"type": kind, "params": params}
    return items

def catalog_ndarray(mod):
    """Catalog ndarray methods and properties."""
    ndarray_cls = getattr(mod, "ndarray", None)
    if ndarray_cls is None:
        return {}, {}

    methods = {}
    properties = {}
    for name in sorted(dir(ndarray_cls)):
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            continue
        try:
            obj = getattr(ndarray_cls, name)
        except Exception:
            continue

        if isinstance(obj, property):
            properties[name] = {"type": "property"}
        elif callable(obj) or isinstance(obj, (classmethod, staticmethod)):
            kind = "operator" if name.startswith("__") else "method"
            params = get_params(obj)
            methods[name] = {"type": kind, "params": params}

    return methods, properties

def main():
    import macmetalpy as mmp

    # Top-level catalog
    top_level = catalog_module(mmp)

    # Submodule catalogs
    submodules = {}
    for submod_name in ["linalg", "fft", "random"]:
        submod = getattr(mmp, submod_name, None)
        if submod is not None:
            submodules[submod_name] = catalog_module(submod, prefix=submod_name)

    # ndarray
    methods, properties = catalog_ndarray(mmp)

    catalog = {
        "top_level": top_level,
        "submodules": submodules,
        "ndarray_methods": methods,
        "ndarray_properties": properties,
    }

    outpath = os.path.join(os.path.dirname(__file__), "macmetalpy_catalog.json")
    with open(outpath, "w") as f:
        json.dump(catalog, f, indent=2, default=str)

    print(f"Cataloged {len(top_level)} top-level items")
    for name, items in submodules.items():
        print(f"  {name}: {len(items)} items")
    print(f"  ndarray methods: {len(methods)}")
    print(f"  ndarray properties: {len(properties)}")
    print(f"Written to {outpath}")

if __name__ == "__main__":
    main()
