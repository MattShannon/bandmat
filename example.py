#!/usr/bin/python
"""A simple example of how the bandmat package may be used."""

# Copyright 2013, 2014, 2015, 2016, 2017 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import numpy as np

import bandmat as bm

a_bm = bm.BandMat(
    1, 1,
    np.array([
        [0.0, 0.2, 0.3, 0.4, 0.5],
        [1.0, 0.9, 1.1, 0.8, 1.3],
        [0.3, 0.1, 0.5, 0.6, 0.0],
    ])
)

b_bm = bm.BandMat(
    1, 0,
    np.array([
        [1.0, 0.9, 1.1, 0.8, 1.3],
        [0.3, 0.1, 0.5, 0.6, 0.0],
    ])
)

c_bm = bm.dot_mm(a_bm.T, b_bm)

d_bm = a_bm + b_bm

a_full = a_bm.full()
b_full = b_bm.full()
c_full = c_bm.full()
d_full = d_bm.full()

print 'a_full:'
print a_full
print
print 'b_full:'
print b_full
print
print 'np.dot(a_full.T, b_full):'
print c_full
print
print 'a_full + b_full:'
print d_full
print

c_full_again = np.dot(a_full.T, b_full)
assert np.allclose(c_full_again, c_full)

d_full_again = a_full + b_full
assert np.allclose(d_full_again, d_full)
