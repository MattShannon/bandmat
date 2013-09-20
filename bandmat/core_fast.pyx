"""Core banded matrix definitions and functions (fast cython code)."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import numpy as np

cimport numpy as np
cimport cython

@cython.boundscheck(False)
def dot_vector_banded(a_lu_tuple,
                      np.ndarray[np.float64_t, ndim=2] a_banded,
                      np.ndarray[np.float64_t, ndim=1] b,
                      long transpose = False,
                      target = None):
    """Multiplies a banded matrix by a vector.

    If transpose is False, then computes A * b.
    If transpose is True, then computes A.T * b.

    If target is specified then the result of the multiplication is added to
    this vector, otherwise a new vector is created and returned.
    """
    # (FIXME : could wrap corresponding BLAS routine (gbmv) instead)
    cdef long frames
    cdef long l_a, u_a
    cdef np.ndarray[np.float64_t, ndim=1] c

    if transpose:
        u_a, l_a = a_lu_tuple
    else:
        l_a, u_a = a_lu_tuple
    assert l_a >= 0 and u_a >= 0 and a_banded.shape[0] == l_a + u_a + 1
    frames = a_banded.shape[1]
    assert b.shape[0] == frames

    if target is not None:
        c = target
    else:
        c = np.zeros((frames,))

    cdef long o_a
    cdef unsigned long row_a
    cdef long d_a
    cdef unsigned long frame

    for o_a in range(-u_a, l_a + 1):
        row_a = (l_a - o_a) if transpose else (u_a + o_a)
        d_a = 0 if transpose else -o_a
        for frame in range(max(0, o_a), max(0, frames + min(0, o_a))):
            c[frame] += (
                a_banded[row_a, frame + d_a] *
                b[frame - o_a]
            )

    if target is None:
        return c
    else:
        return

@cython.boundscheck(False)
def dot_matrix_banded_add(a_rep, b_rep, target_rep,
                          np.ndarray[np.float64_t, ndim=1] diag = None,
                          long transpose_a = False,
                          long transpose_b = False):
    """Banded matrix multiplication (existing target).

    By default computes A * B, where banded matrices A and B are specified by
    a_rep and b_rep, and adds the result to the banded matrix C specified by
    target_rep.
    If diag is specified, computes A * D * B where D is a diagonal matrix
    specified by diag.
    If transpose_a is True, then the transpose of A is used in place of A.
    If transpose_b is True, then the transpose of B is used in place of B.

    If the banded representation of C does not contain enough rows to contain
    the result of A * B, then only diagonals of A * B which contribute to the
    rows present in the banded representation of C are computed.
    This is more efficient in the case that not all diagonals of C are needed.
    """
    cdef long frames
    cdef long l_a, u_a
    cdef long l_b, u_b
    cdef long l_c, u_c
    cdef long use_diag
    cdef np.ndarray[np.float64_t, ndim=2] a_banded
    cdef np.ndarray[np.float64_t, ndim=2] b_banded
    cdef np.ndarray[np.float64_t, ndim=2] c_banded

    if transpose_a:
        u_a, l_a, a_banded = a_rep
    else:
        l_a, u_a, a_banded = a_rep
    if transpose_b:
        u_b, l_b, b_banded = b_rep
    else:
        l_b, u_b, b_banded = b_rep
    l_c, u_c, c_banded = target_rep
    assert l_a >= 0 and u_a >= 0 and a_banded.shape[0] == l_a + u_a + 1
    assert l_b >= 0 and u_b >= 0 and b_banded.shape[0] == l_b + u_b + 1
    assert l_c >= 0 and u_c >= 0 and c_banded.shape[0] == l_c + u_c + 1
    frames = a_banded.shape[1]
    assert b_banded.shape[1] == frames
    assert c_banded.shape[1] == frames
    use_diag = (diag is not None)
    if use_diag:
        assert diag.shape[0] == frames

    cdef long o_a, o_b, o_c
    cdef unsigned long row_a, row_b, row_c
    cdef long d_a, d_b
    cdef unsigned long frame

    for o_c in range(-min(u_c, u_a + u_b), min(l_c, l_a + l_b) + 1):
        for o_a in range(-min(u_a, l_b - o_c), min(l_a, u_b + o_c) + 1):
            o_b = o_c - o_a
            row_a = (l_a - o_a) if transpose_a else (u_a + o_a)
            row_b = (l_b - o_b) if transpose_b else (u_b + o_b)
            d_a = o_a if transpose_a else 0
            d_b = 0 if transpose_b else -o_b
            row_c = u_c + o_a + o_b
            for frame in range(max(0, -o_a, o_b),
                               max(0, frames + min(0, -o_a, o_b))):
                c_banded[row_c, frame - o_b] += (
                    a_banded[row_a, frame + d_a] *
                    b_banded[row_b, frame + d_b] *
                    (diag[frame] if use_diag else 1.0)
                )

    return

@cython.boundscheck(False)
def band_of_outer_add(np.ndarray[np.float64_t, ndim=1] a_vec,
                      np.ndarray[np.float64_t, ndim=1] b_vec,
                      target_rep,
                      double mult = 1.0):
    """Adds the outer product of two vectors to a banded matrix.

    The two vectors are specified by a_vec and b_vec.
    The banded matrix to add to is specified in banded representation by
    target_rep.
    """
    cdef long frames
    cdef long l, u
    cdef np.ndarray[np.float64_t, ndim=2] mat_banded

    frames = a_vec.shape[0]
    assert b_vec.shape[0] == frames
    l, u, mat_banded = target_rep
    assert l >= 0 and u >= 0
    assert mat_banded.shape[0] == l + u + 1
    assert mat_banded.shape[1] == frames

    cdef long o_m
    cdef unsigned long row
    cdef unsigned long frame

    for o_m in range(-u, l + 1):
        row = u + o_m
        for frame in range(max(0, -o_m), max(0, frames + min(0, -o_m))):
            mat_banded[row, frame] += a_vec[frame + o_m] * b_vec[frame] * mult

    return

@cython.boundscheck(False)
def band_of_inverse_from_chol(np.ndarray[np.float64_t, ndim=2] chol_banded):
    """Computes band of the inverse of a positive definite banded matrix.

    Computes band of the inverse of a positive definite banded matrix given its
    Cholesky decomposition.
    Equivalently, finds the band of the covariance matrix of a discrete time
    process which is linear-Gaussian backwards in time.

    chol_banded should be the banded representation of the lower-triangular
    matrix L, where P = L * L.T is the Cholesky decomposition of P.
    """
    cdef unsigned long depth
    cdef long frames
    cdef np.ndarray[np.float64_t, ndim=2] cov_banded

    depth = chol_banded.shape[0] - 1
    frames = chol_banded.shape[1]
    cov_banded = np.zeros((depth * 2 + 1, frames))

    cdef long frame_l
    cdef unsigned long frame
    cdef long curr_depth
    cdef long k_1, k_2
    cdef double mult

    curr_depth = 0
    for frame_l in range(frames - 1, -1, -1):
        frame = frame_l
        mult = 1.0 / chol_banded[<unsigned long>(0), frame]
        cov_banded[depth, frame] = mult * mult
        for k_2 in range(curr_depth, -1, -1):
            for k_1 in range(curr_depth, 0, -1):
                cov_banded[depth + k_2, frame] -= (
                    chol_banded[<unsigned long>(k_1), frame] *
                    cov_banded[depth + k_1 - k_2, frame + k_2] *
                    mult
                )
        for k_2 in range(curr_depth, 0, -1):
            cov_banded[depth - k_2, frame + k_2] = (
                cov_banded[depth + k_2, frame]
            )
        if curr_depth < depth:
            curr_depth += 1

    return cov_banded

# FIXME : move elsewhere (nothing to do with banded matrices)
def plusEquals(np.ndarray[np.int64_t, ndim=1] targetIndexSeq,
               np.ndarray[np.float64_t, ndim=1] source,
               np.ndarray[np.float64_t, ndim=1] target):
    """Implements a += method with fancy indexing.

    Does what you might expect
        target[targetIndexSeq] += source
    to do.
    """
    cdef unsigned long sourceSize

    sourceSize = source.shape[0]
    assert targetIndexSeq.shape[0] == sourceSize

    cdef unsigned long sourceIndex
    cdef long targetIndex

    for sourceIndex in range(sourceSize):
        targetIndex = targetIndexSeq[sourceIndex]
        target[targetIndex] += source[sourceIndex]

    return

# FIXME : move elsewhere (nothing to do with banded matrices)
def plusEquals2D(np.ndarray[np.int64_t, ndim=1] targetIndexSeq,
                 np.ndarray[np.float64_t, ndim=2] source,
                 np.ndarray[np.float64_t, ndim=2] target):
    """Implements a += method with fancy indexing.

    Does what you might expect
        target[targetIndexSeq] += source
    to do.
    """
    cdef unsigned long sourceSize
    cdef unsigned long size1

    sourceSize = source.shape[0]
    assert targetIndexSeq.shape[0] == sourceSize
    size1 = source.shape[1]
    assert target.shape[1] == size1

    cdef unsigned long sourceIndex
    cdef long targetIndex
    cdef unsigned long index1

    for sourceIndex in range(sourceSize):
        targetIndex = targetIndexSeq[sourceIndex]
        for index1 in range(size1):
            target[targetIndex, index1] += source[sourceIndex, index1]

    return
