"""Helper functions for testing."""

import numpy as np
from numpy.random import randn

def assert_allclose(actual, desired, rtol = 1e-7, atol = 1e-14, msg = 'items not almost equal'):
    if np.shape(actual) != np.shape(desired) or not np.allclose(actual, desired, rtol, atol):
        raise AssertionError(msg+'\n ACTUAL:\n'+repr(actual)+'\n DESIRED:\n'+repr(desired))

def assert_allequal(actual, desired, msg = 'items not equal'):
    if np.shape(actual) != np.shape(desired) or not np.all(actual == desired):
        raise AssertionError(msg+'\n ACTUAL:\n'+repr(actual)+'\n DESIRED:\n'+repr(desired))

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
