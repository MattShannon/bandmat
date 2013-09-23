"""Tests for addition, multiplication, etc using banded matrices."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from bandmat.testhelp import assert_allclose
from bandmat.test_core import gen_BandMat

import bandmat as bm
import bandmat.full as fl

import unittest
import numpy as np
import random
from numpy.random import randn, randint

def randBool():
    return randint(0, 2) == 0

class TestTensor(unittest.TestCase):
    def test_plus_equals_band_of(self, its = 100):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            target_bm_orig = gen_BandMat(size)
            mat_bm = gen_BandMat(size)

            target_bm = target_bm_orig.copy_exact()
            bm.plus_equals_band_of(target_bm, mat_bm)

            target_full_good = target_bm_orig.full()
            target_full_good += (
                fl.band_ec(target_bm.l, target_bm.u, mat_bm.full())
            )
            assert_allclose(target_bm.full(), target_full_good)

    def test_plus(self, its = 100):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            b_bm = gen_BandMat(size)

            c_bm = bm.plus(a_bm, b_bm)

            c_full_good = a_bm.full() + b_bm.full()
            assert_allclose(c_bm.full(), c_full_good)

    def test_scalar_mult(self, its = 100):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            mult = randn()
            overwrite = randBool()

            b_full_good = a_bm.full() * mult

            if overwrite:
                bm.scalar_mult(a_bm, mult, overwrite = True)
                b_bm = a_bm
            else:
                b_bm = bm.scalar_mult(a_bm, mult)
            assert b_bm.l == a_bm.l
            assert b_bm.u == a_bm.u
            assert_allclose(b_bm.full(), b_full_good)

    def test_dot_mv_plus_equals(self, its = 100):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            b = randn(size)
            c_orig = randn(size)

            c = c_orig.copy()
            bm.dot_mv_plus_equals(a_bm, b, c)

            c_good = c_orig.copy()
            c_good += np.dot(a_bm.full(), b)
            assert_allclose(c, c_good)

    def test_dot_mv(self, its = 100):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            b = randn(size)

            c = bm.dot_mv(a_bm, b)

            c_good = np.dot(a_bm.full(), b)
            assert_allclose(c, c_good)

    def test_dot_mm_plus_equals(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            b_bm = gen_BandMat(size)
            c_bm_orig = gen_BandMat(size)
            diag = None if randBool() else randn(size)

            c_bm = c_bm_orig.copy_exact()
            bm.dot_mm_plus_equals(a_bm, b_bm, c_bm, diag = diag)

            c_full_good = c_bm_orig.full()
            c_full_good += fl.band_ec(
                c_bm.l, c_bm.u,
                np.dot(
                    np.dot(
                        a_bm.full(),
                        np.diag(np.ones((size,)) if diag is None else diag)
                    ),
                    b_bm.full()
                )
            )
            assert_allclose(c_bm.full(), c_full_good)

    def test_dot_mm(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            b_bm = gen_BandMat(size)
            diag = None if randBool() else randn(size)

            c_bm = bm.dot_mm(a_bm, b_bm, diag = diag)
            assert c_bm.l == a_bm.l + b_bm.l
            assert c_bm.u == a_bm.u + b_bm.u
            assert c_bm.size == size

            c_full_good = np.dot(
                np.dot(
                    a_bm.full(),
                    np.diag(np.ones((size,)) if diag is None else diag)
                ),
                b_bm.full()
            )
            assert_allclose(c_bm.full(), c_full_good)

    def test_dot_mm_partial(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            b_bm = gen_BandMat(size)
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])
            diag = None if randBool() else randn(size)

            c_bm = bm.dot_mm_partial(l, u, a_bm, b_bm, diag = diag)
            assert c_bm.l == l
            assert c_bm.u == u
            assert c_bm.size == size

            c_full_good = fl.band_ec(
                l, u,
                np.dot(
                    np.dot(
                        a_bm.full(),
                        np.diag(np.ones((size,)) if diag is None else diag)
                    ),
                    b_bm.full()
                )
            )
            assert_allclose(c_bm.full(), c_full_good)

    def test_dot_mmm_partial(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            a_bm = gen_BandMat(size)
            b_bm = gen_BandMat(size)
            c_bm = gen_BandMat(size)
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])

            d_bm = bm.dot_mmm_partial(l, u, a_bm, b_bm, c_bm)
            assert d_bm.l == l
            assert d_bm.u == u
            assert d_bm.size == size

            d_full_good = fl.band_ec(
                l, u,
                np.dot(
                    a_bm.full(),
                    np.dot(b_bm.full(), c_bm.full())
                )
            )
            assert_allclose(d_bm.full(), d_full_good)

    def test_band_of_outer_plus_equals(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm_orig = gen_BandMat(size)
            a_vec = randn(size)
            b_vec = randn(size)
            mult = randn()

            mat_bm = mat_bm_orig.copy_exact()
            bm.band_of_outer_plus_equals(a_vec, b_vec, mat_bm, mult = mult)

            mat_full_good = mat_bm_orig.full()
            mat_full_good += fl.band_ec(
                mat_bm.l, mat_bm.u,
                np.outer(a_vec, b_vec) * mult
            )
            assert_allclose(mat_bm.full(), mat_full_good)

if __name__ == '__main__':
    unittest.main()
