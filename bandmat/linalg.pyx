"""Linear algebra operations for banded matrices."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import bandmat as bm
from bandmat import BandMat

import numpy as np
import scipy.linalg as sla

cimport numpy as cnp
cimport cython

cnp.import_array()
cnp.import_ufunc()

def cholesky(mat_bm, lower = False, alternative = False):
    """Computes the Cholesky factor of a positive definite banded matrix.

    The conventional Cholesky decomposition of a positive definite matrix P is
    P = L . L.T, where the lower Cholesky factor L is lower-triangular, the
    upper Cholesky factor L.T is upper-triangular, and . indicates matrix
    multiplication.
    Each positive definite matrix has a unique Cholesky decomposition.
    Given a positive definite banded matrix, this function computes its
    Cholesky factor, which is also a banded matrix.

    This function can also work with the alternative Cholesky decomposition.
    The alternative Cholesky decomposition of P is P = L.T . L, where again L
    is lower-triangular.
    Whereas the conventional Cholesky decomposition is intimately connected
    with linear Gaussian autoregressive probabilistic models defined forwards
    in time, the alternative Cholesky decomposition is intimately connected
    with linear Gaussian autoregressive probabilistic models defined backwards
    in time.

    `mat_bm` (representing P above) should represent the whole matrix, not just
    the lower or upper triangles.
    If the matrix represented by `mat_bm` is not symmetric then behavior of
    this function is undefined (currently either the upper or lower triangle is
    used to compute the Cholesky factor and the rest of the matrix is ignored).
    If `lower` is True then the lower Cholesky factor is returned, and
    otherwise the upper Cholesky factor is returned.
    If `alternative` is True then the Cholesky factor returned is for the
    alternative Cholesky decomposition, and otherwise it is for the
    conventional Cholesky decomposition.
    """
    if mat_bm.transposed:
        mat_bm = mat_bm.T

    depth = mat_bm.l
    assert mat_bm.u == depth
    assert depth >= 0

    l = depth if lower else 0
    u = 0 if lower else depth

    if mat_bm.size == 0:
        chol_bm = BandMat(l, u, np.zeros((depth + 1, 0)))
    else:
        prec_half_data = mat_bm.data[(depth - u):(depth + l + 1)]
        if alternative:
            chol_data = sla.cholesky_banded(prec_half_data[::-1, ::-1],
                                            lower = not lower)[::-1, ::-1]
        else:
            chol_data = sla.cholesky_banded(prec_half_data, lower = lower)
        chol_bm = BandMat(l, u, chol_data)

    return chol_bm

@cython.boundscheck(False)
def band_of_inverse_from_chol(chol_bm):
    """Computes the band of the inverse of a positive definite banded matrix.

    Computes the band of the inverse of a positive definite banded matrix given
    its Cholesky factor.
    Equivalently, finds the band of the covariance matrix of a discrete time
    process which is linear-Gaussian backwards in time.

    `chol_bm` can be either the lower or upper Cholesky factor.
    """
    assert chol_bm.l == 0 or chol_bm.u == 0

    if chol_bm.u != 0:
        chol_bm = chol_bm.T

    cdef unsigned long depth
    cdef long frames
    cdef long l_chol, u_chol, transposed_chol
    cdef cnp.ndarray[cnp.float64_t, ndim=2] chol_data
    cdef cnp.ndarray[cnp.float64_t, ndim=2] cov_data

    l_chol = chol_bm.l
    u_chol = chol_bm.u
    chol_data = chol_bm.data
    transposed_chol = chol_bm.transposed
    assert l_chol >= 0
    assert u_chol == 0
    assert chol_data.shape[0] == l_chol + u_chol + 1

    depth = l_chol
    frames = chol_data.shape[1]
    cov_data = np.zeros((depth * 2 + 1, frames))

    cdef long frame_l
    cdef unsigned long frame
    cdef long curr_depth
    cdef long k_1, k_2
    cdef unsigned long row_chol
    cdef long d_chol
    cdef double mult

    curr_depth = 0
    for frame_l in range(frames - 1, -1, -1):
        frame = frame_l
        row_chol = l_chol if transposed_chol else u_chol
        mult = 1.0 / chol_data[row_chol, frame]
        cov_data[depth, frame] = mult * mult
        for k_2 in range(curr_depth, -1, -1):
            for k_1 in range(curr_depth, 0, -1):
                row_chol = l_chol - k_1 if transposed_chol else u_chol + k_1
                d_chol = k_1 if transposed_chol else 0
                cov_data[depth + k_2, frame] -= (
                    chol_data[row_chol, frame + d_chol] *
                    cov_data[depth + k_1 - k_2, frame + k_2] *
                    mult
                )
        for k_2 in range(curr_depth, 0, -1):
            cov_data[depth - k_2, frame + k_2] = (
                cov_data[depth + k_2, frame]
            )
        if curr_depth < depth:
            curr_depth += 1

    cov_bm = BandMat(depth, depth, cov_data)
    return cov_bm

def band_of_inverse(mat_bm):
    """Computes band of the inverse of a positive definite banded matrix."""
    depth = mat_bm.l
    assert mat_bm.u == depth
    chol_bm = cholesky(mat_bm, lower = True)
    band_of_inv_bm = band_of_inverse_from_chol(chol_bm)
    return band_of_inv_bm
