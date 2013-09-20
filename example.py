#!/usr/bin/python
"""A simple example of how the bandmat package may be used."""

import numpy as np
import bandmat as bm

a_banded = np.array([
    [0.0, 0.2, 0.3, 0.4, 0.5],
    [1.0, 0.9, 1.1, 0.8, 1.3],
    [0.3, 0.1, 0.5, 0.6, 0.0],
])
l_a = 1
u_a = 1

b_banded = np.array([
    [1.0, 0.9, 1.1, 0.8, 1.3],
    [0.3, 0.1, 0.5, 0.6, 0.0],
])
l_b = 1
u_b = 0

l_c, u_c, c_banded = bm.dot_matrix_banded(
    (l_a, u_a, a_banded),
    (l_b, u_b, b_banded),
    transpose_a = True
)

a_full = bm.unbandize((l_a, u_a), a_banded)
b_full = bm.unbandize((l_b, u_b), b_banded)
c_full = bm.unbandize((l_c, u_c), c_banded)

print 'a_full:'
print a_full
print
print 'b_full:'
print b_full
print
print 'np.dot(a_full.T, b_full):'
print c_full
print

c_full_again = np.dot(a_full.T, b_full)
assert np.allclose(c_full_again, c_full)
