bandmat
=======

|CI status|

.. |CI status| image:: https://github.com/MattShannon/bandmat/actions/workflows/ci.yml/badge.svg?branch=master
   :alt: CI status
   :target: https://github.com/MattShannon/bandmat/actions/workflows/ci.yml

This package provides a simple banded matrix library for python.
It supports banded matrix-vector and matrix-matrix multiplication, converting
between full and banded matrix representations, and certain linear algebra
operations on banded matrices.
It builds on the excellent numpy and scipy packages, which have limited support
for banded matrix operations.

Overview
--------

A banded matrix is a matrix where only the diagonal, a number of superdiagonals
and a number of subdiagonals are non-zero.
The well-known BLAS interface and LAPACK library for linear algebra define
several banded matrix operations, and some of these, such as banded Cholesky
decomposition, are wrapped in the excellent python package scipy, specifically
in scipy.linalg.
The bandmat package re-uses the banded matrix representation used by BLAS,
LAPACK and scipy.linalg, wrapping it in a lightweight class for ease of use.
See the docstring for the BandMat class for full details of the representation
used.

The bandmat package provides:

- a lightweight class wrapping the LAPACK-style banded matrix representation.
  This class keeps track of things like bandwidths to allow a more direct
  coding style when working with banded matrices.
- some basic banded matrix operations not present in scipy.
  For example, banded matrix-vector multiplication is defined by BLAS but not
  wrapped by scipy, and banded matrix-matrix multiplication is not defined in
  BLAS or in scipy.
  The bandmat package contains C implementations of these operations written in
  cython.
- helper functions for converting between full and banded matrix
  representations.
- certain linear algebra operations on banded matrices, including computing the
  band of the inverse of a banded matrix.

Only square banded matrices are supported by this package.

License
-------

Please see the file ``License`` for details of the license and warranty for
bandmat.

Installation
------------

For most purposes the simplest way to install bandmat is to use pip in a
virtual environment::

    python3 -m venv env
    env/bin/pip install bandmat

This installs the latest released version of
`bandmat on PyPI <https://pypi.python.org/pypi/bandmat>`_, together with any
currently uninstalled python packages required by bandmat.

The latest development version of bandmat is available from a github repository
(see below).

To check that bandmat is installed correctly you can run the test suite::

    python -m unittest discover bandmat

Examples
--------

See the package docstring (run ``import bandmat as bm; help(bm)`` in the python
interpreter) for some examples of usage.
The python script ``example.py`` also contains some simple examples of usage.
To run it::

    python example.py

Development
-----------

The source code is hosted in the
`bandmat github repository <https://github.com/MattShannon/bandmat>`_.
To obtain the latest source code using git::

    git clone git://github.com/MattShannon/bandmat.git

To install bandmat for development in a virtual environment::

    python3 -m venv env
    env/bin/pip install -e .

This will install the build dependencies (cython, numpy) and compile the
cython extensions automatically.
To recompile after modifying ``.pyx`` files::

    python setup.py build_ext --inplace

Bugs
----

Please use the
`issue tracker <https://github.com/MattShannon/bandmat/issues>`_ to submit bug
reports.

Contact
-------

The author of bandmat is `Matt Shannon <mailto:matt.shannon@cantab.net>`_.
