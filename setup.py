#!/usr/bin/python
"""A setuptools-based script for distributing and installing bandmat."""

# Copyright 2013, 2014, 2015, 2016, 2017 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import os
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.build_ext import build_ext as _build_ext

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

requires = [line.rstrip('\n') for line in open('requirements.txt')]

src_ext = '.c'
cmdclass = {}

# see "A note on setup.py" in README.rst for an explanation of the dev file
dev_mode = os.path.exists('dev')

if dev_mode:
    src_ext = '.pyx'


    class Sdist(_sdist):
        def __init__(self, *args, **kwargs):
            from Cython.Build import cythonize
            _sdist.__init__(self, *args, **kwargs)

        """A cythonizing sdist command.

        This class is a custom sdist command which ensures all cython-generated
        C files are up-to-date before running the conventional sdist command.
        """
        def run(self):
            cythonize([os.path.join(*loc) + src_ext for loc in cython_locs])
            _sdist.run(self)

    cmdclass['sdist'] = Sdist


class BuildExt(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process
        __builtins__.__NUMPY_SETUP__ = False

        import numpy as np
        self.include_dirs.append(np.get_include())


cmdclass['build_ext'] = BuildExt

ext_modules = [
    Extension('.'.join(loc), [os.path.join(*loc) + src_ext],
              extra_compile_args=['-Wno-unused-but-set-variable', '-O3'])
    for loc in cython_locs
]

setup(
    name='bandmat',
    version='0.7.dev1',
    description='A banded matrix library for python.',
    url='http://github.com/MattShannon/bandmat',
    author='Matt Shannon',
    author_email='matt.shannon@cantab.net',
    license='3-clause BSD (see License file)',
    packages=['bandmat'],
    setup_requires=[requires[0:2]],
    install_requires=requires,
    long_description=long_description,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
