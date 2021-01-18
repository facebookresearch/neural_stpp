# Copyright (c) Facebook, Inc. and its affiliates.

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# Run "python setup.py build_ext --inplace"
setup(
    ext_modules=cythonize("data_utils_fast.pyx"),
    include_dirs=[np.get_include()],
)
