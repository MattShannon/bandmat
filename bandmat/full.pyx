"""Banded operations on full matrices.

More precisely, this module provides operations involving the bands of square
matrices that are stored using the conventional numpy matrix representation.

The (l, u)-extra entries of a rectangular matrix `mat_rect` are defined as the
entries which have no effect on the result of `band_c(l, u, mat_rect)`.
They lie in the upper-left and bottom-right corners of `mat_rect`.
"""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import numpy as np

# FIXME : write more efficient (and arguably simpler?) cython implementation?
def band_e(l, u, mat_full):
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
    size = len(mat_full)
    assert np.shape(mat_full) == (size, size)

    mat_rect = np.zeros((l + u + 1, size))
    mat_rect[u] = np.diag(mat_full)
    for offset in range(1, u + 1):
        if offset <= size:
            mat_rect[u - offset, offset:size] = np.diag(mat_full, offset)
    for offset in range(1, l + 1):
        if offset <= size:
            mat_rect[u + offset, 0:(size - offset)] = np.diag(mat_full, -offset)

    return mat_rect

# FIXME : write much more efficient cython implementation
def band_c(l, u, mat_rect):
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
    size = np.shape(mat_rect)[1]
    assert np.shape(mat_rect) == (l + u + 1, size)

    mat_full = np.diag(mat_rect[u])
    for offset in range(1, u + 1):
        if offset <= size:
            # (FIXME : could make this more efficient)
            mat_full += np.diag(mat_rect[u - offset, offset:size], offset)
    for offset in range(1, l + 1):
        if offset <= size:
            # (FIXME : could make this more efficient)
            mat_full += np.diag(mat_rect[u + offset, 0:(size - offset)],
                                -offset)

    return mat_full

def zero_extra_entries(l, u, mat_rect):
    """Zeroes the extra entries of a rectangular matrix.

    Equivalent to:

        mat_rect[:] = band_e(l, u, band_c(l, u, mat_rect))

    N.B. in-place, i.e. mutates `mat_rect`.
    """
    assert l >= 0
    assert u >= 0
    size = np.shape(mat_rect)[1]
    assert np.shape(mat_rect) == (l + u + 1, size)

    for offset in range(1, u + 1):
        mat_rect[u - offset, 0:min(offset, size)] = 0.0
    for offset in range(1, l + 1):
        mat_rect[u + offset, max(size - offset, 0):size] = 0.0

def band_ce(l, u, mat_rect):
    """Copies a rectangular matrix and zeroes its extra entries."""
    mat_rect_new = mat_rect.copy()
    zero_extra_entries(l, u, mat_rect_new)
    return mat_rect_new

def band_ec(l, u, mat_full):
    """Copies a square matrix and zeroes entries outside a given band."""
    return band_c(l, u, band_e(l, u, mat_full))
