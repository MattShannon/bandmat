#!/usr/bin/python
"""A setuptools-based script for distributing and installing bandmat."""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

cython_locs = [
    ('bandmat', 'full'),
    ('bandmat', 'core'),
    ('bandmat', 'tensor'),
    ('bandmat', 'linalg'),
    ('bandmat', 'misc'),
    ('bandmat', 'overlap'),
]

with open('README.rst') as readme_file:
    long_description = readme_file.read()

requires = [ line.rstrip('\n') for line in open('requirements.txt') ]

ext_modules = cythonize([
    Extension('.'.join(loc), ['/'.join(loc)+'.pyx'],
              extra_compile_args=['-O3'],
              include_dirs=[np.get_include()])
    for loc in cython_locs
])

setup(
    name='bandmat',
    version='0.8.dev1',
    description='A banded matrix library for python.',
    url='http://github.com/MattShannon/bandmat',
    author='Matt Shannon',
    author_email='matt.shannon@cantab.net',
    license='3-clause BSD (see License file)',
    packages=['bandmat'],
    install_requires=requires,
    long_description=long_description,
    ext_modules=ext_modules,
)
