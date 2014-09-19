"""Functions to do with overlapping subtensors."""

# Copyright 2013, 2014 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import bandmat as bm

import numpy as np

cimport numpy as cnp
cimport cython

cnp.import_array()
cnp.import_ufunc()

@cython.boundscheck(False)
def sum_overlapping_v(cnp.ndarray[cnp.float64_t, ndim=2] contribs):
    """Computes the overlapped sum of a sequence of vectors.

    If `contribs` has shape (T, K + 1) then the returned vector has size T + K.
    The value K here must be non-negative.

    The overlapped sum of a sequence of vectors is defined as follows.
    Suppose the vectors in `contribs` are "laid out" along some larger vector
    such that each element of `contribs` is offset by 1 index relative to the
    previous element.
    For example `contribs[0]` occupies the left edge of the larger vector and
    `contribs[1]` is positioned 1 index "right" of this.
    The overlapped sum is the sum of these vectors laid out in this way.
    """
    assert contribs.shape[1] >= 1

    cdef unsigned long depth = contribs.shape[1] - 1
    cdef unsigned long width = depth + 1
    cdef unsigned long size = contribs.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vec = (
        np.zeros((size + depth,), dtype=contribs.dtype)
    )

    cdef unsigned long frame, k

    for frame in range(size):
        for k in range(width):
            vec[frame + k] += contribs[frame, k]

    return vec

@cython.boundscheck(False)
def sum_overlapping_m(cnp.ndarray[cnp.float64_t, ndim=3] contribs):
    """Computes the overlapped sum of a sequence of square matrices.

    If `contribs` has shape (T, K + 1, K + 1) then the overlapped sum is a
    square banded matrix of size T + K, and this is returned as a BandMat with
    upper and lower bandwidth K.
    The value K here must be non-negative.

    The overlapped sum of a sequence of matrices is defined as follows.
    Suppose the matrices in `contribs` are "laid out" along the diagonal of
    some larger matrix such that each element of `contribs` is 1 index further
    down and 1 index further right than the previous element.
    For example `contribs[0]` occupies the top left corner of the larger matrix
    and `contribs[1]` is 1 index down and right of this.
    The overlapped sum is the sum of these matrices laid out in this way.
    """
    assert contribs.shape[1] >= 1
    assert contribs.shape[2] == contribs.shape[1]

    cdef unsigned long depth = contribs.shape[1] - 1
    cdef unsigned long width = depth + 1
    cdef unsigned long size = contribs.shape[0]
    mat_bm = bm.zeros(depth, depth, size + depth)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_data = mat_bm.data

    cdef unsigned long frame, k, l

    for frame in range(size):
        for k in range(width):
            for l in range(width):
                mat_data[depth + k - l, frame + l] += contribs[frame, k, l]

    return mat_bm

@cython.boundscheck(False)
def extract_overlapping_v(cnp.ndarray[cnp.float64_t, ndim=1] vec,
                          unsigned long depth):
    """Extracts overlapping subvectors from a vector.

    If `vec` has shape (T,) where K is equal to `depth` then a matrix of shape
    (T + K, K + 1) is returned.
    The returned matrix is a sequence of subvectors of `vec`.
    """
    cdef unsigned long width = depth + 1
    cdef unsigned long size = vec.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] subvectors = (
        np.empty((size + depth, width), dtype=vec.dtype)
    )

    cdef unsigned long frame, k

    for frame in range(0, depth):
        for k in range(width):
            if frame + k >= depth and frame + k < size + depth:
                subvectors[frame, k] = vec[frame + k - depth]
            else:
                subvectors[frame, k] = 0.0

    for frame in range(depth, size):
        for k in range(width):
            subvectors[frame, k] = vec[frame + k - depth]

    for frame in range(size, size + depth):
        for k in range(width):
            if frame + k >= depth and frame + k < size + depth:
                subvectors[frame, k] = vec[frame + k - depth]
            else:
                subvectors[frame, k] = 0.0

    return subvectors

@cython.boundscheck(False)
def extract_overlapping_m(mat_bm):
    """Extracts overlapping submatrices along the diagonal of a banded matrix.

    If `mat_bm` has size T and upper and lower bandwidth K then a tensor of
    shape (T + K, K + 1, K + 1) is returned.
    The returned tensor is a sequence of submatrices from along the diagonal of
    the matrix represented by `mat_bm`.
    """
    assert mat_bm.l == mat_bm.u

    cdef unsigned long depth = mat_bm.l
    cdef unsigned long width = depth + 1
    cdef unsigned long size = mat_bm.size
    cdef unsigned long transposed = mat_bm.transposed
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_data = mat_bm.data
    cdef cnp.ndarray[cnp.float64_t, ndim=3] submats = (
        np.empty((size + depth, width, width), dtype=mat_data.dtype)
    )

    cdef unsigned long frame, k, l

    if transposed:
        for frame in range(0, depth):
            for k in range(width):
                if frame + k >= depth and frame + k < size + depth:
                    for l in range(width):
                        if frame + l >= depth and frame + l < size + depth:
                            submats[frame, l, k] = (
                                mat_data[depth + k - l, frame + l - depth]
                            )
                        else:
                            submats[frame, l, k] = 0.0
                else:
                    for l in range(width):
                        submats[frame, l, k] = 0.0

        for frame in range(depth, size):
            for k in range(width):
                for l in range(width):
                    submats[frame, l, k] = (
                        mat_data[depth + k - l, frame + l - depth]
                    )

        for frame in range(size, size + depth):
            for k in range(width):
                if frame + k >= depth and frame + k < size + depth:
                    for l in range(width):
                        if frame + l >= depth and frame + l < size + depth:
                            submats[frame, l, k] = (
                                mat_data[depth + k - l, frame + l - depth]
                            )
                        else:
                            submats[frame, l, k] = 0.0
                else:
                    for l in range(width):
                        submats[frame, l, k] = 0.0

    else:
        for frame in range(0, depth):
            for k in range(width):
                if frame + k >= depth and frame + k < size + depth:
                    for l in range(width):
                        if frame + l >= depth and frame + l < size + depth:
                            submats[frame, k, l] = (
                                mat_data[depth + k - l, frame + l - depth]
                            )
                        else:
                            submats[frame, k, l] = 0.0
                else:
                    for l in range(width):
                        submats[frame, k, l] = 0.0

        for frame in range(depth, size):
            for k in range(width):
                for l in range(width):
                    submats[frame, k, l] = (
                        mat_data[depth + k - l, frame + l - depth]
                    )

        for frame in range(size, size + depth):
            for k in range(width):
                if frame + k >= depth and frame + k < size + depth:
                    for l in range(width):
                        if frame + l >= depth and frame + l < size + depth:
                            submats[frame, k, l] = (
                                mat_data[depth + k - l, frame + l - depth]
                            )
                        else:
                            submats[frame, k, l] = 0.0
                else:
                    for l in range(width):
                        submats[frame, k, l] = 0.0

    return submats
