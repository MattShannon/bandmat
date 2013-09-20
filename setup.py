#!/usr/bin/python
"""A distutils-based script for distributing and installing bandmat."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

with open('README.rst') as readmeFile:
    long_description = readmeFile.read()

ext_modules = [
    Extension('bandmat.core_fast',
              [os.path.join('bandmat', 'core_fast.pyx')]),
]

setup(
    name = 'bandmat',
    version = '0.1.dev1',
    description = 'A banded matrix library for python.',
    url = 'http://github.com/MattShannon/bandmat',
    author = 'Matt Shannon',
    author_email = 'matt.shannon@cantab.net',
    license = '3-clause BSD (see License file)',
    packages = ['bandmat'],
    long_description = long_description,
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
)
