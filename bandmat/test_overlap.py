"""Tests for functions to do with overlapping subtensors."""

# Copyright 2013, 2014 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from bandmat.testhelp import assert_allclose, assert_allequal
from bandmat.test_core import gen_BandMat

import bandmat as bm
import bandmat.overlap as bmo

import unittest
import numpy as np
import random
from numpy.random import randn, randint

cc = bm.band_e_bm_common

class TestOverlap(unittest.TestCase):
    def test_sum_overlapping_v(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            contribs = randn(size, depth + 1)

            vec = bmo.sum_overlapping_v(contribs)
            assert vec.shape == (size + depth,)

            if size == 0:
                # check action for no contributions
                assert_allequal(vec, np.zeros((depth,)))
            elif size == 1:
                # check action for a single contribution
                assert_allequal(vec, contribs[0])
            else:
                # check action under splitting list of contributions in two
                splitPos = randint(size + 1)
                vec_again = np.zeros((size + depth,))
                vec_again[0:(splitPos + depth)] += (
                    bmo.sum_overlapping_v(contribs[:splitPos])
                )
                vec_again[splitPos:(size + depth)] += (
                    bmo.sum_overlapping_v(contribs[splitPos:])
                )
                assert_allclose(vec, vec_again)

    def test_sum_overlapping_m(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            contribs = randn(size, depth + 1, depth + 1)

            mat_bm = bmo.sum_overlapping_m(contribs)

            assert mat_bm.size == size + depth
            assert mat_bm.l == mat_bm.u == depth

            if size == 0:
                # check action for no contributions
                assert_allequal(mat_bm.full(), np.zeros((depth, depth)))
            elif size == 1:
                # check action for a single contribution
                assert_allequal(mat_bm.full(), contribs[0])
            else:
                # check action under splitting list of contributions in two
                splitPos = randint(size + 1)
                mat_bm_again = bm.zeros(depth, depth, size + depth)
                mat_bm_again.sub_matrix_view(0, splitPos + depth).__iadd__(
                    bmo.sum_overlapping_m(contribs[:splitPos])
                )
                mat_bm_again.sub_matrix_view(splitPos, size + depth).__iadd__(
                    bmo.sum_overlapping_m(contribs[splitPos:])
                )
                assert_allclose(*cc(mat_bm, mat_bm_again))

    def test_extract_overlapping_v(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            vec = randn(size + depth)

            subvectors = bmo.extract_overlapping_v(vec, depth)
            assert subvectors.shape == (size, depth + 1)
            for frame in range(size):
                assert_allequal(
                    subvectors[frame],
                    vec[frame:(frame + depth + 1)]
                )

    def test_extract_overlapping_m(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            mat_bm = gen_BandMat(size + depth, l=depth, u=depth)
            mat_full = mat_bm.full()

            submats = bmo.extract_overlapping_m(mat_bm)
            assert submats.shape == (size, depth + 1, depth + 1)
            for frame in range(size):
                assert_allequal(
                    submats[frame],
                    mat_full[frame:(frame + depth + 1),
                             frame:(frame + depth + 1)]
                )

if __name__ == '__main__':
    unittest.main()
