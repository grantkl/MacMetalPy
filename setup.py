from setuptools import setup, Extension
import numpy as np

ext = Extension(
    "macmetalpy._accelerator",
    sources=["src/macmetalpy/_accelerator.c"],
    include_dirs=[np.get_include()],
)

setup(ext_modules=[ext])
