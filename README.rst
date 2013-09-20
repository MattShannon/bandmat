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
  The bandmat package contains C implementations of these operations written in
  cython.
- helper functions for converting between full and banded matrix
  representations.
- certain linear algebra operations on banded matrices.
  Currently the only supported operation is finding the band of the inverse of
  a positive definite banded matrix.
- an implementation of a fancy indexed += function for numpy arrays.
  This is included for (the author's) convenience and is not specific to banded
  matrices.

License
-------

Please see the file ``License`` for details of the license and warranty for
bandmat.

Development
-----------

The source code is hosted in the
`bandmat github repository <https://github.com/MattShannon/bandmat>`_.
To obtain the latest source code using git::

    git clone git://github.com/MattShannon/bandmat.git

Development is in fact done using `darcs <http://darcs.net/>`_, with the darcs
repository converted to a git repository using
`darcs-to-git <https://github.com/purcell/darcs-to-git>`_.

To compile the cython part of bandmat in the current directory::

    python setup.py build_ext --inplace

This command must be run after every modification to the source ``.pyx`` files.

A note on ``setup.py``
----------------------

The included ``setup.py`` file operates in one of two modes depending on
whether or not the file ``dev`` is present in the project root directory.
In development mode (``dev`` present, as for the github repository), the
``build_ext`` command uses cython to compile cython modules from their ``.pyx``
source, and the ``sdist`` command is modified to first use cython to compile
cython modules from their ``.pyx`` source to ``.c`` files.
In distribution mode (``dev`` absent, as for source distributions), the
``build_ext`` command uses a C compiler to directly compile
cython modules from the corresponding ``.c`` files.
This approach ensures that source distributions can be installed on systems
without cython or with an incompatible version of cython, while ensuring that
distributed ``.c`` files are always up-to-date and that the source ``.pyx``
files are used instead of ``.c`` files during development.

The author would welcome any suggestions for more elegant ways to achieve a
similar effect to the approach described above!

Bugs
----

Please use the
`issue tracker <https://github.com/MattShannon/bandmat/issues>`_ to submit bug
reports.

Contact
-------

The author of bandmat is `Matt Shannon <mailto:matt.shannon@cantab.net>`_.
