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
def sum_overlapping_v(cnp.ndarray[cnp.float64_t, ndim=2] contribs,
                      cnp.ndarray[cnp.float64_t, ndim=1] target=None):
    """Computes the overlapped sum of a sequence of vectors.

    The overlapped sum of a sequence of vectors is defined as follows.
    Suppose the vectors in `contribs` are "laid out" along some larger vector
    such that each element of `contribs` is offset by 1 index relative to the
    previous element.
    For example `contribs[0]` occupies the left edge of the larger vector and
    `contribs[1]` is positioned 1 index "right" of this.
    The overlapped sum is the sum of these vectors laid out in this way, which
    is a vector.

    If `target` is None then a new vector is returned; otherwise the overlapped
    sum is added to the vector `target`.
    If `contribs` has shape (T, K + 1) then the returned vector (or `target` if
    specified) has size T + K.
    The value K here must be non-negative.
    """
    assert contribs.shape[1] >= 1

    cdef unsigned long depth = contribs.shape[1] - 1
    cdef unsigned long width = depth + 1
    cdef unsigned long size = contribs.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vec

    if target is None:
        vec = np.zeros((size + depth,), dtype=contribs.dtype)
    else:
        assert target.shape[0] == size + depth
        vec = target

    cdef unsigned long frame, k

    for frame in range(size):
        for k in range(width):
            vec[frame + k] += contribs[frame, k]

    if target is None:
        return vec
    else:
        return

@cython.boundscheck(False)
def sum_overlapping_m(cnp.ndarray[cnp.float64_t, ndim=3] contribs,
                      target_bm=None):
    """Computes the overlapped sum of a sequence of square matrices.

    The overlapped sum of a sequence of matrices is defined as follows.
    Suppose the matrices in `contribs` are "laid out" along the diagonal of
    some larger matrix such that each element of `contribs` is 1 index further
    down and 1 index further right than the previous element.
    For example `contribs[0]` occupies the top left corner of the larger matrix
    and `contribs[1]` is 1 index down and right of this.
    The overlapped sum is the sum of these matrices laid out in this way, which
    is a banded matrix.
    For this function the contributions are square, so the resulting banded
    matrix is also square.

    If `target_bm` is None then a new BandMat is returned; otherwise the
    overlapped sum is added to the BandMat `target_bm`.
    If `contribs` has shape (T, K + 1, K + 1) then the returned BandMat (or
    `target_bm` if specified) has size T + K and upper and lower bandwidth K.
    The value K here must be non-negative.
    """
    assert contribs.shape[1] >= 1
    assert contribs.shape[2] == contribs.shape[1]

    cdef unsigned long depth = contribs.shape[1] - 1
    cdef unsigned long width = depth + 1
    cdef unsigned long size = contribs.shape[0]

    if target_bm is None:
        mat_bm = bm.zeros(depth, depth, size + depth)
    else:
        assert target_bm.l == depth and target_bm.u == depth
        assert target_bm.size == size + depth
        mat_bm = target_bm

    cdef unsigned long transposed = mat_bm.transposed
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_data = mat_bm.data

    cdef unsigned long frame, k, l

    if transposed:
        for frame in range(size):
            for k in range(width):
                for l in range(width):
                    mat_data[depth + l - k, frame + k] += contribs[frame, k, l]
    else:
        for frame in range(size):
            for k in range(width):
                for l in range(width):
                    mat_data[depth + k - l, frame + l] += contribs[frame, k, l]

    if target_bm is None:
        return mat_bm
    else:
        return

@cython.boundscheck(False)
def extract_overlapping_v(cnp.ndarray[cnp.float64_t, ndim=1] vec,
                          unsigned long depth,
                          cnp.ndarray[cnp.float64_t, ndim=2] target=None):
    """Extracts overlapping subvectors from a vector.

    The result `subvectors` is a matrix consisting of a sequence of subvectors
    of `vec`.
    Specifically `subvectors[i]` is `vec[i:(i + depth + 1)]`.

    If `target` is None then a new matrix is returned; otherwise the result is
    written to the matrix `target` (and all elements of `target` are guaranteed
    to be overwritten, so there is no need to zero it ahead of time).
    If `vec` has shape (T + K,) where K is equal to `depth` then the returned
    matrix (or `target` if specified) has shape (T, K + 1).
    """
    assert vec.shape[0] >= depth

    cdef unsigned long width = depth + 1
    cdef unsigned long size = vec.shape[0] - depth
    cdef cnp.ndarray[cnp.float64_t, ndim=2] subvectors

    if target is None:
        subvectors = np.empty((size, width), dtype=vec.dtype)
    else:
        assert target.shape[0] == size
        assert target.shape[1] == width
        subvectors = target

    cdef unsigned long frame, k

    for frame in range(size):
        for k in range(width):
            subvectors[frame, k] = vec[frame + k]

    if target is None:
        return subvectors
    else:
        return

@cython.boundscheck(False)
def extract_overlapping_m(mat_bm,
                          cnp.ndarray[cnp.float64_t, ndim=3] target=None):
    """Extracts overlapping submatrices along the diagonal of a banded matrix.

    The result `submats` is rank-3 tensor consisting of a sequence of
    submatrices from along the diagonal of the matrix represented by `mat_bm`.
    If `mat_full` is the matrix represented by `mat_bm` and `depth` is K above
    then `submats[i]` is `mat_full[i:(i + depth + 1), i:(i + depth + 1)]`.

    If `target` is None then a new tensor is returned; otherwise the result is
    written to the tensor `target` (and all elements of `target` are guaranteed
    to be overwritten, so there is no need to zero it ahead of time).
    If `mat_bm` has size T + K and upper and lower bandwidth K then the
    returned tensor (or `target` is specified) has shape (T, K + 1, K + 1).
    """
    assert mat_bm.l == mat_bm.u
    assert mat_bm.size >= mat_bm.l

    cdef unsigned long depth = mat_bm.l
    cdef unsigned long width = depth + 1
    cdef unsigned long size = mat_bm.size - depth
    cdef unsigned long transposed = mat_bm.transposed
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat_data = mat_bm.data
    cdef cnp.ndarray[cnp.float64_t, ndim=3] submats

    if target is None:
        submats = np.empty((size, width, width), dtype=mat_data.dtype)
    else:
        assert target.shape[0] == size
        assert target.shape[1] == width
        assert target.shape[2] == width
        submats = target

    cdef unsigned long frame, k, l

    if transposed:
        for frame in range(size):
            for k in range(width):
                for l in range(width):
                    submats[frame, k, l] = mat_data[depth + l - k, frame + k]
    else:
        for frame in range(size):
            for k in range(width):
                for l in range(width):
                    submats[frame, k, l] = mat_data[depth + k - l, frame + l]

    if target is None:
        return submats
    else:
        return

def sum_overlapping_v_chunked(contribs_chunks, depth, target):
    """A chunked version of sum_overlapping_v.

    The elements of the iterator `contribs_chunks` should be of the form
    `(start, end, contribs)`.
    For example if `contribs_all` has length 10 then

        sum_overlapping_v_chunked([
            (0, 3, contribs_all[0:3]),
            (3, 10, contribs_all[3:10])
        ])

    produces the same result as `sum_overlapping_v(contribs_all)`.
    This can be used to construct more memory-efficient code in some cases
    (though not in the above example).
    """
    assert depth >= 0
    size = len(target) - depth
    assert size >= 0

    for start, end, contribs in contribs_chunks:
        sum_overlapping_v(contribs, target=target[start:(end + depth)])

def sum_overlapping_m_chunked(contribs_chunks, target_bm):
    """A chunked version of sum_overlapping_m.

    The elements of the iterator `contribs_chunks` should be of the form
    `(start, end, contribs)`.
    For example if `contribs_all` has length 10 then

        sum_overlapping_m_chunked([
            (0, 3, contribs_all[0:3]),
            (3, 10, contribs_all[3:10])
        ])

    produces the same result as `sum_overlapping_m(contribs_all)`.
    This can be used to construct more memory-efficient code in some cases
    (though not in the above example).
    """
    depth = target_bm.l
    assert target_bm.u == depth
    size = target_bm.size - depth
    assert size >= 0

    for start, end, contribs in contribs_chunks:
        sum_overlapping_m(
            contribs,
            target_bm=target_bm.sub_matrix_view(start, end + depth)
        )

def extract_overlapping_v_chunked(vec, depth, chunk_size):
    """A chunked version of extract_overlapping_v.

    An iterator over chunks of the output of extract_overlapping_v is returned.
    This can be used to construct more memory-efficient code in some cases.
    """
    size = len(vec) - depth
    assert size >= 0
    assert chunk_size >= depth + 1

    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)
        subvectors = extract_overlapping_v(
            vec[start:(end + depth)],
            depth
        )
        yield start, end, subvectors

def extract_overlapping_m_chunked(mat_bm, chunk_size):
    """A chunked version of extract_overlapping_m.

    An iterator over chunks of the output of extract_overlapping_m is returned.
    This can be used to construct more memory-efficient code in some cases.
    """
    depth = mat_bm.l
    assert mat_bm.u == depth
    size = mat_bm.size - depth
    assert size >= 0
    assert chunk_size >= depth + 1

    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)
        submats = extract_overlapping_m(
            mat_bm.sub_matrix_view(start, end + depth)
        )
        yield start, end, submats
