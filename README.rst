bandmat
=======

This package provides a simple banded matrix library for python.
It supports banded matrix-vector and matrix-matrix multiplication, converting
between full and banded matrix representations, and certain linear algebra
operations on banded matrices.

Overview
--------

The well-known BLAS interface and LAPACK library for linear algebra define
several banded matrix operations, and some of these, such as banded Cholesky
decomposition, are wrapped in the excellent python package scipy, specifically
in scipy.linalg.
The bandmat package re-uses the banded matrix representation used by BLAS,
LAPACK and scipy.linalg.
This represents a square n by n banded matrix as a d by n numpy array together
with the lower or sub-diagonal bandwidth l and the upper or super-diagonal
bandwidth u, where the overall bandwidth d is equal to l + u + 1.
Only square banded matrices are supported by this package.
Full details of the banded representation can be found in the docstring for
scipy.linalg.solve_banded.

The bandmat package provides:

  - some basic banded matrix operations not present in scipy.
    For example, banded matrix-vector multiplication is defined by BLAS but not
    wrapped by scipy, and banded matrix-matrix multiplication is not defined in
    BLAS or in scipy.
    The bandmat package contains C implementations of these operations written
    in cython.
  - helper functions for converting between full and banded matrix
    representations.
  - certain linear algebra operations on banded matrices.
    Currently the only supported operation is finding the band of the inverse
    of a positive definite banded matrix.
  - an implementation of a fancy indexed += function for numpy arrays.
    This is included for (the author's) convenience and is not specific to
    banded matrices.

Set-up
------

The latest version of bandmat is available from a github repository (see
below).
Required packages are numpy, scipy and cython.

To compile the cython part of bandmat in the current directory::

    python setup.py build_ext --inplace

License
-------

Please see the file ``License`` for details of the license and warranty for
bandmat.

Source
------

The source code is hosted in the
`bandmat github repository <https://github.com/MattShannon/bandmat>`_.
To obtain the latest source code using git::

    git clone git://github.com/MattShannon/bandmat.git

Development is in fact done using `darcs <http://darcs.net/>`_, with the darcs
repository converted to a git repository using
`darcs-to-git <https://github.com/purcell/darcs-to-git>`_.

Bugs
----

Please use the
`issue tracker <https://github.com/MattShannon/bandmat/issues>`_ to submit bug
reports.

Contact
-------

The author of bandmat is `Matt Shannon <mailto:matt.shannon@cantab.net>`_.
