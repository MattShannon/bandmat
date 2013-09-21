"""Helper functions for testing."""

import numpy as np

def assert_allclose(actual, desired, rtol = 1e-7, atol = 1e-14, msg = 'items not almost equal'):
    if np.shape(actual) != np.shape(desired) or not np.allclose(actual, desired, rtol, atol):
        raise AssertionError(msg+'\n ACTUAL:\n'+repr(actual)+'\n DESIRED:\n'+repr(desired))

def assert_allequal(actual, desired, msg = 'items not equal'):
    if np.shape(actual) != np.shape(desired) or not np.all(actual == desired):
        raise AssertionError(msg+'\n ACTUAL:\n'+repr(actual)+'\n DESIRED:\n'+repr(desired))
