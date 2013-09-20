
# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from bandmat.core_fast import dot_matrix_banded_add
from bandmat.core_fast import band_of_inverse_from_chol

import numpy as np
import scipy.linalg as sla

def bandize((l, u), a, checkNextUpperDiagZero = True, checkNextLowerDiagZero = True):
    """Extract band-diagonal of a square matrix.

    Extracts diagonal of a together with u super-diagonals and l sub-diagonals
    and returns result in (l, u) banded storage format.

    Banded storage format means the format used by scipy.linalg.solve_banded.
    """
    assert l >= 0
    assert u >= 0
    size = len(a)
    assert np.shape(a) == (size, size)
    a_banded = np.zeros((u + 1 + l, size))
    a_banded[u] = np.diag(a)
    for offset in range(1, u + 1):
        if offset <= size:
            a_banded[u - offset, offset:size] = np.diag(a, offset)
    for offset in range(1, l + 1):
        if offset <= size:
            a_banded[u + offset, 0:(size - offset)] = np.diag(a, -offset)
    if checkNextUpperDiagZero:
        assert all(np.abs(np.diag(a, u + 1)) < 1e-12)
    if checkNextLowerDiagZero:
        # N.B. below check necessary to workaround numpy bug where np.transpose
        #   and np.diag interact badly for out-of-range subdiagonals, e.g.
        #   np.diag(np.transpose(np.eye(3)), -4) should give [] array but
        #   actually gives [1.0] array with numpy version 1.5.1 (numpy 1.3.0
        #   has correct behaviour).
        if l + 1 <= size:
            assert all(np.abs(np.diag(a, -(l + 1))) < 1e-12)
    return a_banded

def unbandize((l, u), a_banded):
    """Constructs a full matrix from a matrix in banded storage format.

    Banded storage format means the format used by scipy.linalg.solve_banded.
    """
    size = np.shape(a_banded)[1]
    assert np.shape(a_banded) == (u + l + 1, size)
    a = np.diag(a_banded[u])
    for offset in range(1, u + 1):
        if offset <= size:
            # (FIXME : could make this more efficient)
            a += np.diag(a_banded[u - offset, offset:size], offset)
    for offset in range(1, l + 1):
        if offset <= size:
            # (FIXME : could make this more efficient)
            a += np.diag(a_banded[u + offset, 0:(size - offset)], -offset)
    return a

def zeroBandedOutsideMatrix((l, u), a_banded):
    """Zeros entries of a banded representation that lie outside the represented matrix.

    (N.B. in-place, i.e. mutates a_banded)
    """
    size = np.shape(a_banded)[1]
    assert np.shape(a_banded) == (u + l + 1, size)

    for offset in range(1, u + 1):
        a_banded[u - offset, 0:min(offset, size)] = 0.0
    for offset in range(1, l + 1):
        a_banded[u + offset, max(size - offset, 0):size] = 0.0

def transpose_banded((l, u), a_banded):
    """Computes transpose of a banded matrix.

    a_banded should be in (l, u) banded storage format.
    Return value is in (u, l) banded storage format.

    Banded storage format means the format used by scipy.linalg.solve_banded.
    """
    assert l >= 0
    assert u >= 0
    size = np.shape(a_banded)[1]
    assert np.shape(a_banded) == (u + 1 + l, size)
    b_banded = np.zeros(np.shape(a_banded))
    b_banded[l] = a_banded[u]
    for offset in range(1, u + 1):
        if offset <= size:
            b_banded[l + offset, 0:(size - offset)] = a_banded[u - offset, offset:size]
    for offset in range(1, l + 1):
        if offset <= size:
            b_banded[l - offset, offset:size] = a_banded[u + offset, 0:(size - offset)]
    return b_banded

def symmetrize_banded((l, u), a_banded_half):
    """Reconstructs symmetric banded matrix from lower or upper portion."""
    assert l >= 0
    assert u >= 0
    if l != 0 and u != 0:
        raise RuntimeError('matrix to symmetrize must either be lower portion only or upper portion only')
    subDiagWidth = max(l, u)
    frames = np.shape(a_banded_half)[1]
    assert np.shape(a_banded_half) == (subDiagWidth + 1, frames)

    a_banded = np.zeros((2 * subDiagWidth + 1, frames))
    if l != 0:
        a_banded[:(subDiagWidth + 1)] = transpose_banded((subDiagWidth, 0), a_banded_half)
        a_banded[subDiagWidth:] = a_banded_half
    else:
        a_banded[subDiagWidth:] = transpose_banded((0, subDiagWidth), a_banded_half)
        a_banded[:(subDiagWidth + 1)] = a_banded_half

    return a_banded

def dot_matrix_banded(a_rep, b_rep,
                      diag = None,
                      transpose_a = False,
                      transpose_b = False):
    """Banded matrix multiplication.

    By default computes A * B, where banded matrices A and B are specified by
    a_rep and b_rep.
    If diag is specified, computes A * D * B where D is a diagonal matrix
    specified by diag.
    If transpose_a is True, then the transpose of A is used in place of A.
    If transpose_b is True, then the transpose of B is used in place of B.
    """
    if transpose_a:
        u_a, l_a, a_banded = a_rep
    else:
        l_a, u_a, a_banded = a_rep
    if transpose_b:
        u_b, l_b, b_banded = b_rep
    else:
        l_b, u_b, b_banded = b_rep
    assert l_a >= 0 and u_a >= 0
    assert l_b >= 0 and u_b >= 0
    frames = np.shape(a_banded)[1]

    l_c = l_a + l_b
    u_c = u_a + u_b
    c_banded = np.zeros((l_c + u_c + 1, frames))

    dot_matrix_banded_add(a_rep, b_rep, (l_c, u_c, c_banded),
                          diag = diag,
                          transpose_a = transpose_a,
                          transpose_b = transpose_b)
    return l_c, u_c, c_banded

def dot_matrix_banded_partial(a_rep, b_rep, result_lu_tuple,
                              diag = None,
                              transpose_a = False,
                              transpose_b = False):
    """Banded matrix multiplication (partial computation of target).

    By default computes A * B, where banded matrices A and B are specified by
    a_rep and b_rep.
    If diag is specified, computes A * D * B where D is a diagonal matrix
    specified by diag.
    If transpose_a is True, then the transpose of A is used in place of A.
    If transpose_b is True, then the transpose of B is used in place of B.

    This function only computes the rows of the banded representation of the
    result that are specified by result_lu_tuple.
    This is more efficient in the case that not all diagonals of the result are
    needed.
    """
    if transpose_a:
        u_a, l_a, a_banded = a_rep
    else:
        l_a, u_a, a_banded = a_rep
    if transpose_b:
        u_b, l_b, b_banded = b_rep
    else:
        l_b, u_b, b_banded = b_rep
    assert l_a >= 0 and u_a >= 0
    assert l_b >= 0 and u_b >= 0
    frames = np.shape(a_banded)[1]

    l_c, u_c = result_lu_tuple
    c_banded = np.zeros((l_c + u_c + 1, frames))

    dot_matrix_banded_add(a_rep, b_rep, (l_c, u_c, c_banded),
                          diag = diag,
                          transpose_a = transpose_a,
                          transpose_b = transpose_b)
    return c_banded

def band_of_inverse(prec_banded):
    """Computes band of the inverse of a positive definite banded matrix."""
    assert len(prec_banded) % 2 == 1
    depth = (len(prec_banded) - 1) // 2
    chol_banded = sla.cholesky_banded(prec_banded[depth:], lower = True)
    cov_banded = band_of_inverse_from_chol(chol_banded)
    return cov_banded
