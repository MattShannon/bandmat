"""Assorted helpful functions."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

cimport numpy as cnp
cimport cython

cnp.import_array()
cnp.import_ufunc()

def fancy_plus_equals(cnp.ndarray[cnp.int64_t, ndim=1] targetIndexSeq,
                      cnp.ndarray[cnp.float64_t, ndim=1] source,
                      cnp.ndarray[cnp.float64_t, ndim=1] target):
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

def fancy_plus_equals_2d(cnp.ndarray[cnp.int64_t, ndim=1] targetIndexSeq,
                         cnp.ndarray[cnp.float64_t, ndim=2] source,
                         cnp.ndarray[cnp.float64_t, ndim=2] target):
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
