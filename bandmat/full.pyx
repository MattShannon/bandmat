"""Banded operations on full matrices.

This module provides operations involving the bands of square matrices that are
stored using the conventional numpy matrix representation.

The (l, u)-extra entries of a rectangular matrix `mat_rect` are defined as the
entries which have no effect on the result of `band_c(l, u, mat_rect)`.
They lie in the upper-left and bottom-right corners of `mat_rect`.
"""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import numpy as np

cimport numpy as cnp
cimport cython

cnp.import_array()
cnp.import_ufunc()

@cython.boundscheck(False)
def band_e(long l, long u, cnp.ndarray[cnp.float64_t, ndim=2] mat_full):
    """Extracts the band of a full, square, banded matrix.

    Given a square numpy matrix `mat_full`, returns a rectangular numpy matrix
    `mat_rect` with rows corresponding to the `u` superdiagonals, the diagonal
    and the `l` subdiagonals of `mat_full`.
    The full matrix is "collapsed column-wise", i.e. each column of `mat_rect`
    contains the same entries as the part of the corresponding column of
    `mat_full` that lies within the band.
    The extra entries in the corners of `mat_rect` which do not correspond to
    any entry in `mat_full` are set to zero.
    """
    assert l >= 0
    assert u >= 0
    assert mat_full.shape[1] == mat_full.shape[0]

    cdef long size
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_rect

    size = mat_full.shape[0]
    mat_rect = np.zeros((l + u + 1, size))

    cdef long i
    cdef unsigned long row
    cdef unsigned long j

    for i in range(-u, l + 1):
        row = u + i
        for j in range(max(0, -i), max(0, size + min(0, -i))):
            mat_rect[row, j] = mat_full[j + i, j]

    return mat_rect

@cython.boundscheck(False)
def band_c(long l, long u, cnp.ndarray[cnp.float64_t, ndim=2] mat_rect):
    """Constructs a full, square, banded matrix from its band.

    Given a rectangular numpy matrix `mat_rect`, returns a square numpy matrix
    `mat_full` with its `u` superdiagonals, diagonal and `l` subdiagonals given
    given by the rows of `mat_rect`.
    The part of each column of `mat_full` that lies within the band contains
    the same entries as the corresponding column of `mat_rect`.
    The extra entries in the corners of `mat_rect` which do not correspond to
    any entry in `mat_full` are ignored.
    """
    assert l >= 0
    assert u >= 0
    assert mat_rect.shape[0] == l + u + 1

    cdef long size
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_full

    size = mat_rect.shape[1]
    mat_full = np.zeros((size, size))

    cdef long i
    cdef unsigned long row
    cdef unsigned long j

    for i in range(-u, l + 1):
        row = u + i
        for j in range(max(0, -i), max(0, size + min(0, -i))):
            mat_full[j + i, j] = mat_rect[row, j]

    return mat_full

@cython.boundscheck(False)
def zero_extra_entries(long l, long u,
                       cnp.ndarray[cnp.float64_t, ndim=2] mat_rect):
    """Zeroes the extra entries of a rectangular matrix.

    Equivalent to:

        mat_rect[:] = band_e(l, u, band_c(l, u, mat_rect))

    N.B. in-place, i.e. mutates `mat_rect`.
    """
    assert l >= 0
    assert u >= 0
    assert mat_rect.shape[0] == l + u + 1

    cdef long size

    size = mat_rect.shape[1]

    cdef long i
    cdef unsigned long row
    cdef unsigned long j

    for i in range(-u, 0):
        row = u + i
        for j in range(0, min(size, -i)):
            mat_rect[row, j] = 0.0
    for i in range(1, l + 1):
        row = u + i
        for j in range(max(0, size - i), size):
            mat_rect[row, j] = 0.0

    return

def band_ce(l, u, mat_rect):
    """Copies a rectangular matrix and zeroes its extra entries."""
    mat_rect_new = mat_rect.copy()
    zero_extra_entries(l, u, mat_rect_new)
    return mat_rect_new

def band_ec(l, u, mat_full):
    """Copies a square matrix and zeroes entries outside a given band."""
    return band_c(l, u, band_e(l, u, mat_full))
