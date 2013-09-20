#!/usr/bin/python
"""A distutils-based script for distributing and installing bandmat."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension('bandmat.core_fast',
              [os.path.join('bandmat', 'core_fast.pyx')]),
]

setup(
    name = 'bandmat',
    description = 'A banded matrix library for python.',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
)
