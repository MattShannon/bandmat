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

def rand_bool():
    return randint(0, 2) == 0

class TestOverlap(unittest.TestCase):
    def test_sum_overlapping_v(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            contribs = randn(size, depth + 1)
            target = randn(size + depth)
            target_orig = target.copy()

            vec = bmo.sum_overlapping_v(contribs)
            assert vec.shape == (size + depth,)

            # check target-based version adds to target correctly
            bmo.sum_overlapping_v(contribs, target=target)
            assert_allclose(target, target_orig + vec)

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
                bmo.sum_overlapping_v(
                    contribs[:splitPos],
                    target=vec_again[0:(splitPos + depth)]
                )
                bmo.sum_overlapping_v(
                    contribs[splitPos:],
                    target=vec_again[splitPos:(size + depth)]
                )
                assert_allclose(vec, vec_again)

    def test_sum_overlapping_m(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            contribs = randn(size, depth + 1, depth + 1)
            target_bm = gen_BandMat(size + depth, l=depth, u=depth)
            target_bm_orig = target_bm.copy()

            mat_bm = bmo.sum_overlapping_m(contribs)
            assert mat_bm.size == size + depth
            assert mat_bm.l == mat_bm.u == depth

            # check target-based version adds to target_bm correctly
            bmo.sum_overlapping_m(contribs, target_bm=target_bm)
            assert_allclose(*cc(target_bm, target_bm_orig + mat_bm))

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
                bmo.sum_overlapping_m(
                    contribs[:splitPos],
                    target_bm=mat_bm_again.sub_matrix_view(0, splitPos + depth)
                )
                bmo.sum_overlapping_m(
                    contribs[splitPos:],
                    target_bm=mat_bm_again.sub_matrix_view(
                        splitPos, size + depth
                    )
                )
                assert_allclose(*cc(mat_bm, mat_bm_again))

    def test_extract_overlapping_v(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            vec = randn(size + depth)
            target = None if rand_bool() else randn(size, depth + 1)

            if target is None:
                subvectors = bmo.extract_overlapping_v(vec, depth)
                assert subvectors.shape == (size, depth + 1)
            else:
                bmo.extract_overlapping_v(vec, depth, target=target)
                subvectors = target

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
            target = None if rand_bool() else randn(size, depth + 1, depth + 1)

            if target is None:
                submats = bmo.extract_overlapping_m(mat_bm)
                assert submats.shape == (size, depth + 1, depth + 1)
            else:
                bmo.extract_overlapping_m(mat_bm, target=target)
                submats = target

            for frame in range(size):
                assert_allequal(
                    submats[frame],
                    mat_bm.sub_matrix_view(frame, frame + depth + 1).full()
                )

if __name__ == '__main__':
    unittest.main()
