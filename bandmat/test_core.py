"""Tests for core banded matrix definitions and functions."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from bandmat.testhelp import assert_allequal

import bandmat as bm
import bandmat.full as fl

import unittest
import doctest
import numpy as np
import random
from numpy.random import randn, randint

def randBool():
    return randint(0, 2) == 0

def gen_BandMat(size, l = None, u = None, transposed = None):
    """Generates a random BandMat."""
    if l is None:
        l = random.choice([0, 1, randint(0, 10)])
    if u is None:
        u = random.choice([0, 1, randint(0, 10)])
    data = randn(l + u + 1, size)
    if transposed is None:
        transposed = randBool()
    return bm.BandMat(l, u, data, transposed = transposed)

# package-level docstring tests (N.B. includes other modules, not just core)
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(bm))
    return tests

class TestCore(unittest.TestCase):
    def test_BandMat_basic(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)

            assert a_bm.size == size

    def test_BandMat_full(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            l, u = a_bm.l, a_bm.u

            # N.B. these tests are not really testing much of anything (they
            #   are virtually identical to the implementation of BandMat.full),
            #   but this is not that surprising since the lines below are kind
            #   of the definition of the representation used by BandMat in the
            #   two cases (transposed True and transposed False).
            if a_bm.transposed:
                assert_allequal(a_bm.full().T, fl.band_c(u, l, a_bm.data))
            else:
                assert_allequal(a_bm.full(), fl.band_c(l, u, a_bm.data))

    def test_BandMat_T(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)

            assert_allequal(a_bm.T.full(), a_bm.full().T)

    def test_BandMat_copy_exact(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm = gen_BandMat(size)
            mat_full_orig = mat_bm.full().copy()

            mat_bm_new = mat_bm.copy_exact()
            assert mat_bm_new.l == mat_bm.l
            assert mat_bm_new.u == mat_bm.u
            assert mat_bm_new.transposed == mat_bm.transposed

            # check that copy represents the same matrix
            assert_allequal(mat_bm_new.full(), mat_full_orig)

            # check that mutating the copy does not change the original
            mat_bm_new.data += 1.0
            assert_allequal(mat_bm.full(), mat_full_orig)

    def test_BandMat_copy(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm = gen_BandMat(size)
            mat_full_orig = mat_bm.full().copy()

            mat_bm_new = mat_bm.copy()
            assert mat_bm_new.l == mat_bm.l
            assert mat_bm_new.u == mat_bm.u
            assert not mat_bm_new.transposed

            # check that copy represents the same matrix
            assert_allequal(mat_bm_new.full(), mat_full_orig)

            # check that mutating the copy does not change the original
            mat_bm_new.data += 1.0
            assert_allequal(mat_bm.full(), mat_full_orig)

    def test_BandMat_equiv(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm = gen_BandMat(size)
            l_new = random.choice([None, 0, 1, randint(0, 10)])
            u_new = random.choice([None, 0, 1, randint(0, 10)])
            transposed_new = random.choice([None, True, False])
            zero_extra = randBool()

            l_new_value = mat_bm.l if l_new is None else l_new
            u_new_value = mat_bm.u if u_new is None else u_new
            transposed_new_value = (mat_bm.transposed if transposed_new is None
                                    else transposed_new)

            if l_new_value < mat_bm.l or u_new_value < mat_bm.u:
                self.assertRaises(AssertionError,
                                  mat_bm.equiv,
                                  l_new = l_new, u_new = u_new,
                                  transposed_new = transposed_new,
                                  zero_extra = zero_extra)
            else:
                mat_bm_new = mat_bm.equiv(l_new = l_new, u_new = u_new,
                                          transposed_new = transposed_new,
                                          zero_extra = zero_extra)
                assert mat_bm_new.l == l_new_value
                assert mat_bm_new.u == u_new_value
                assert mat_bm_new.transposed == transposed_new_value
                assert_allequal(mat_bm_new.full(), mat_bm.full())
                if zero_extra:
                    mat_new_data_good = (
                        fl.band_e(u_new_value, l_new_value, mat_bm.full().T)
                    ) if mat_bm_new.transposed else (
                        fl.band_e(l_new_value, u_new_value, mat_bm.full())
                    )
                    assert_allequal(mat_bm_new.data, mat_new_data_good)

    def test_zeros(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])

            mat_bm = bm.zeros(l, u, size)
            assert mat_bm.l == l
            assert mat_bm.u == u
            assert_allequal(mat_bm.full(), np.zeros((size, size)))

    def test_from_full(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            mat_full = gen_BandMat(size).full()
            zero_outside_band = np.all(fl.band_ec(l, u, mat_full) == mat_full)

            if zero_outside_band:
                mat_bm = bm.from_full(l, u, mat_full)
                assert mat_bm.l == l
                assert mat_bm.u == u
                assert_allequal(mat_bm.full(), mat_full)
            else:
                self.assertRaises(AssertionError, bm.from_full, l, u, mat_full)

    def test_band_c_bm(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            mat_rect = randn(l + u + 1, size)

            mat_bm = bm.band_c_bm(l, u, mat_rect)

            mat_full_good = fl.band_c(l, u, mat_rect)
            assert_allequal(mat_bm.full(), mat_full_good)

    def test_band_e_bm(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm = gen_BandMat(size)
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])

            mat_rect = bm.band_e_bm(l, u, mat_bm)

            mat_rect_good = fl.band_e(l, u, mat_bm.full())
            assert_allequal(mat_rect, mat_rect_good)

    def test_band_ec_bm_view(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])

            b_bm = bm.band_ec_bm_view(l, u, a_bm)

            b_full_good = fl.band_ec(l, u, a_bm.full())
            assert_allequal(b_bm.full(), b_full_good)

    def test_band_ec_bm(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])

            b_bm = bm.band_ec_bm(l, u, a_bm)

            b_full_good = fl.band_ec(l, u, a_bm.full())
            assert_allequal(b_bm.full(), b_full_good)

if __name__ == '__main__':
    unittest.main()
