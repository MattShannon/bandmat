"""Helper functions for testing."""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import numpy as np
from numpy.random import randn

def assert_allclose(actual, desired, rtol=1e-7, atol=1e-14,
                    msg='items not almost equal'):
    if np.shape(actual) != np.shape(desired):
        raise AssertionError(
            f'{msg} (wrong shape)\n ACTUAL:  {actual!r}\n DESIRED: {desired!r}'
        )
    if not np.allclose(actual, desired, rtol, atol):
        abs_err = np.abs(actual - desired)
        rel_err = np.abs((actual - desired) / desired)
        raise AssertionError(
            f'{msg}\n ACTUAL:\n{actual!r}\n DESIRED:\n{desired!r}\n'
            f' ABS ERR: {abs_err!r} (max {np.max(abs_err)})\n'
            f' REL ERR: {rel_err!r} (max {np.max(rel_err)})'
        )

def assert_allequal(actual, desired, msg='items not equal'):
    if np.shape(actual) != np.shape(desired):
        raise AssertionError(
            f'{msg} (wrong shape)\n ACTUAL:  {actual!r}\n DESIRED: {desired!r}'
        )
    if not np.all(actual == desired):
        raise AssertionError(
            f'{msg}\n ACTUAL:\n{actual!r}\n DESIRED:\n{desired!r}'
        )

def randomize_extra_entries(l, u, mat_rect):
    """Randomizes the extra entries of a rectangular matrix.

    See the docstring for `band_c` for the definition of extra entries.

    N.B. in-place, i.e. mutates `mat_rect`.
    """
    assert l >= 0
    assert u >= 0
    assert mat_rect.shape[0] == l + u + 1

    size = mat_rect.shape[1]

    for offset in range(1, u + 1):
        mat_rect[u - offset, 0:min(offset, size)] = randn()
    for offset in range(1, l + 1):
        mat_rect[u + offset, max(size - offset, 0):size] = randn()

def randomize_extra_entries_bm(mat_bm):
    if mat_bm.transposed:
        randomize_extra_entries(mat_bm.u, mat_bm.l, mat_bm.data)
    else:
        randomize_extra_entries(mat_bm.l, mat_bm.u, mat_bm.data)

def get_array_mem(*arrays):
    """Returns a representation of the memory layout of an array.

    This is intended to be used to check whether the memory used by a given
    numpy array, or how this memory is mapped into the logical indices of the
    tensor it represents, changes between two points in time.

    Example usage:
    >>> import numpy as np
    >>> x = np.array([2.0, 3.0, 4.0])
    >>> array_mem = get_array_mem(x)
    >>> # some potentially complicated operation
    >>> x *= 2.0
    >>> assert get_array_mem(x) == array_mem
    """
    mems = []
    for array in arrays:
        d = dict(array.__array_interface__)
        if d['strides'] is None:
            d['strides'] = array.strides
        mems.append(d)
    return mems
