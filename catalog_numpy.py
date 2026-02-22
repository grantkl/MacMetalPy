#!/usr/bin/env python3
"""Catalog all public APIs in numpy, writing JSON output."""

import importlib
import inspect
import json
import os
import numpy as np

def get_params(obj):
    """Get parameter names for a callable, or None if uninspectable."""
    try:
        sig = inspect.signature(obj)
        return list(sig.parameters.keys())
    except (ValueError, TypeError):
        return None

def get_signature_str(obj):
    """Get a string representation of the signature."""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return None

def catalog_module(mod):
    """Return dict of {name: {type, params, signature}} for all public items."""
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
        elif isinstance(obj, np.ufunc):
            kind = "ufunc"
        elif callable(obj):
            kind = "function"
        else:
            kind = "constant"

        params = None
        sig_str = None
        if kind in ("function", "ufunc"):
            params = get_params(obj)
            sig_str = get_signature_str(obj)

        items[name] = {"type": kind, "params": params, "signature": sig_str}
    return items

def catalog_ndarray():
    """Catalog numpy ndarray methods and properties."""
    methods = {}
    properties = {}
    for name in sorted(dir(np.ndarray)):
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            continue
        try:
            obj = getattr(np.ndarray, name)
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
    # Top-level catalog
    top_level = catalog_module(np)

    # Submodule catalogs
    submodules = {}
    for submod_name in ["linalg", "fft", "random"]:
        submod = getattr(np, submod_name, None)
        if submod is not None:
            submodules[submod_name] = catalog_module(submod)

    # ndarray
    methods, properties = catalog_ndarray()

    catalog = {
        "numpy_version": np.__version__,
        "top_level": top_level,
        "submodules": submodules,
        "ndarray_methods": methods,
        "ndarray_properties": properties,
    }

    outpath = os.path.join(os.path.dirname(__file__), "numpy_catalog.json")
    with open(outpath, "w") as f:
        json.dump(catalog, f, indent=2, default=str)

    print(f"NumPy version: {np.__version__}")
    print(f"Cataloged {len(top_level)} top-level items")
    for name, items in submodules.items():
        print(f"  {name}: {len(items)} items")
    print(f"  ndarray methods: {len(methods)}")
    print(f"  ndarray properties: {len(properties)}")
    print(f"Written to {outpath}")

if __name__ == "__main__":
    main()
