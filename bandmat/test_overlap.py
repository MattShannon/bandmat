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

def chunk_randomly(xs):
    size = len(xs)

    num_divs = random.choice([0, randint(size // 2 + 1), randint(size + 3)])
    divs = [0] + sorted(
        [ randint(size + 1) for _ in range(num_divs) ]
    ) + [size]

    for start, end in zip(divs, divs[1:]):
        yield start, end, xs[start:end]

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
                split_pos = randint(size + 1)
                vec_again = np.zeros((size + depth,))
                bmo.sum_overlapping_v(
                    contribs[:split_pos],
                    target=vec_again[0:(split_pos + depth)]
                )
                bmo.sum_overlapping_v(
                    contribs[split_pos:],
                    target=vec_again[split_pos:(size + depth)]
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
                split_pos = randint(size + 1)
                mat_bm_again = bm.zeros(depth, depth, size + depth)
                bmo.sum_overlapping_m(
                    contribs[:split_pos],
                    target_bm=mat_bm_again.sub_matrix_view(0, split_pos + depth)
                )
                bmo.sum_overlapping_m(
                    contribs[split_pos:],
                    target_bm=mat_bm_again.sub_matrix_view(
                        split_pos, size + depth
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

    def test_sum_overlapping_v_chunked(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            contribs = randn(size, depth + 1)
            contribs_chunks = chunk_randomly(contribs)
            target = randn(size + depth)
            target_orig = target.copy()

            bmo.sum_overlapping_v_chunked(contribs_chunks, depth, target)
            vec_good = bmo.sum_overlapping_v(contribs)
            assert_allclose(target, target_orig + vec_good)

    def test_sum_overlapping_m_chunked(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            contribs = randn(size, depth + 1, depth + 1)
            contribs_chunks = chunk_randomly(contribs)
            target_bm = gen_BandMat(size + depth, l=depth, u=depth)
            target_bm_orig = target_bm.copy()

            bmo.sum_overlapping_m_chunked(contribs_chunks, target_bm)
            mat_bm_good = bmo.sum_overlapping_m(contribs)
            assert_allclose(*cc(target_bm, target_bm_orig + mat_bm_good))

    def test_extract_overlapping_v_chunked(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            vec = randn(size + depth)
            chunk_size = depth + random.choice([1, randint(1, 10)])

            indices_remaining = set(range(size))
            subvectors_all = np.empty((size, depth + 1))
            for start, end, subvectors in bmo.extract_overlapping_v_chunked(
                vec, depth, chunk_size
            ):
                assert end >= start + 1
                for index in range(start, end):
                    assert index in indices_remaining
                    indices_remaining.remove(index)
                subvectors_all[start:end] = subvectors

            subvectors_good = bmo.extract_overlapping_v(vec, depth)
            assert_allclose(subvectors_all, subvectors_good)

    def test_extract_overlapping_m_chunked(self, its=50):
        for it in range(its):
            size = random.choice([0, 1, randint(10), randint(100)])
            depth = random.choice([0, 1, randint(0, 10)])
            mat_bm = gen_BandMat(size + depth, l=depth, u=depth)
            chunk_size = depth + random.choice([1, randint(1, 10)])

            indices_remaining = set(range(size))
            submats_all = np.empty((size, depth + 1, depth + 1))
            for start, end, submats in bmo.extract_overlapping_m_chunked(
                mat_bm, chunk_size
            ):
                assert end >= start + 1
                for index in range(start, end):
                    assert index in indices_remaining
                    indices_remaining.remove(index)
                submats_all[start:end] = submats

            submats_good = bmo.extract_overlapping_m(mat_bm)
            assert_allclose(submats_all, submats_good)

if __name__ == '__main__':
    unittest.main()
