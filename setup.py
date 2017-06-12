#!/usr/bin/python
"""A setuptools-based script for distributing and installing bandmat."""

# Copyright 2013, 2014, 2015, 2016, 2017 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import os
import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.sdist import sdist as _sdist

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

# see "A note on setup.py" in README.rst for an explanation of the dev file
dev_mode = os.path.exists('dev')

if dev_mode:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize

    class sdist(_sdist):
        """A cythonizing sdist command.

        This class is a custom sdist command which ensures all cython-generated
        C files are up-to-date before running the conventional sdist command.
        """
        def run(self):
            cythonize([ os.path.join(*loc)+'.pyx' for loc in cython_locs ])
            _sdist.run(self)

    cmdclass = {'build_ext': build_ext, 'sdist': sdist}
    ext_modules = [
        Extension('.'.join(loc), [os.path.join(*loc)+'.pyx'],
                  extra_compile_args=['-Wno-unused-but-set-variable', '-O3'],
                  include_dirs=[np.get_include()])
        for loc in cython_locs
    ]
else:
    cmdclass = {}
    ext_modules = [
        Extension('.'.join(loc), [os.path.join(*loc)+'.c'],
                  extra_compile_args=['-Wno-unused-but-set-variable', '-O3'],
                  include_dirs=[np.get_include()])
        for loc in cython_locs
    ]

setup(
    name='bandmat',
    version='0.6',
    description='A banded matrix library for python.',
    url='http://github.com/MattShannon/bandmat',
    author='Matt Shannon',
    author_email='matt.shannon@cantab.net',
    license='3-clause BSD (see License file)',
    packages=['bandmat'],
    install_requires=requires,
    long_description=long_description,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
