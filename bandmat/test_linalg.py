"""Tests for linear algebra operations for banded matrices."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from bandmat.testhelp import assert_allclose, randomize_extra_entries_bm
from bandmat.test_core import gen_BandMat

import bandmat as bm
import bandmat.full as fl
import bandmat.linalg as bla

import unittest
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import random
from numpy.random import randn, randint

def randBool():
    return randint(0, 2) == 0

def gen_symmetric_BandMat(size, depth = None):
    if depth is None:
        depth = random.choice([0, 1, randint(0, 10)])
    a_bm = gen_BandMat(size, l = depth, u = depth)
    b_bm = a_bm + a_bm.T
    randomize_extra_entries_bm(b_bm)
    return b_bm

def gen_pos_def_BandMat(size, depth = None, contribRank = 2):
    """Generates a random positive definite BandMat."""
    assert contribRank >= 0
    if depth is None:
        depth = random.choice([0, 1, randint(0, 10)])
    mat_bm = bm.zeros(depth, depth, size)
    for _ in range(contribRank):
        diff = randint(0, depth + 1)
        chol_bm = gen_BandMat(size, l = depth - diff, u = diff)
        bm.dot_mm_plus_equals(chol_bm, chol_bm.T, mat_bm)
    transposed = randBool()
    if transposed:
        mat_bm = mat_bm.T
    randomize_extra_entries_bm(mat_bm)
    return mat_bm

def gen_chol_factor_BandMat(size, depth = None, contribRank = 2):
    """Generates a random Cholesky factor BandMat.

    This works by generating a random positive definite matrix and then
    computing its Cholesky factor, since using a random matrix as a Cholesky
    factor seems to often lead to ill-conditioned matrices.
    """
    mat_bm = gen_pos_def_BandMat(size, depth = depth,
                                 contribRank = contribRank)
    chol_bm = bla.cholesky(mat_bm, lower = randBool())
    if randBool():
        chol_bm = chol_bm.T
    assert chol_bm.l == 0 or chol_bm.u == 0
    assert chol_bm.l + chol_bm.u == mat_bm.l
    randomize_extra_entries_bm(chol_bm)
    return chol_bm

class TestLinAlg(unittest.TestCase):
    def test_cholesky(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm = gen_pos_def_BandMat(size)
            depth = mat_bm.l
            lower = randBool()
            alternative = randBool()

            chol_bm = bla.cholesky(mat_bm, lower = lower,
                                   alternative = alternative)
            assert chol_bm.l == (depth if lower else 0)
            assert chol_bm.u == (0 if lower else depth)
            assert not np.may_share_memory(chol_bm.data, mat_bm.data)

            if lower != alternative:
                mat_bm_again = bm.dot_mm(chol_bm, chol_bm.T)
            else:
                mat_bm_again = bm.dot_mm(chol_bm.T, chol_bm)
            assert_allclose(mat_bm_again.full(), mat_bm.full())

    def test_cho_solve(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            b = randn(size)
            chol_bm = gen_chol_factor_BandMat(size)
            depth = chol_bm.l + chol_bm.u
            lower = (chol_bm.u == 0)
            chol_full = chol_bm.full()

            x = bla.cho_solve(chol_bm, b)
            if size == 0:
                x_good = np.zeros((size,))
            else:
                x_good = sla.cho_solve((chol_full, lower), b)
            assert_allclose(x, x_good)
            assert not np.may_share_memory(x, chol_bm.data)
            assert not np.may_share_memory(x, b)

    def test_solve(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            b = randn(size)
            # the below tries to ensure the matrix is well-conditioned
            a_bm = gen_BandMat(size) + bm.diag(np.ones((size,)) * 10.0)
            a_full = a_bm.full()

            x = bla.solve(a_bm, b)
            if size == 0:
                x_good = np.zeros((size,))
            else:
                x_good = sla.solve(a_full, b)
            assert_allclose(x, x_good)
            assert not np.may_share_memory(x, a_bm.data)
            assert not np.may_share_memory(x, b)

    def test_solveh(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            b = randn(size)
            a_bm = gen_pos_def_BandMat(size)
            a_full = a_bm.full()

            x = bla.solveh(a_bm, b)
            if size == 0:
                x_good = np.zeros((size,))
            else:
                x_good = sla.solve(a_full, b, sym_pos = True)
            assert_allclose(x, x_good)
            assert not np.may_share_memory(x, a_bm.data)
            assert not np.may_share_memory(x, b)

    def test_band_of_inverse_from_chol(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            chol_bm = gen_chol_factor_BandMat(size)
            depth = chol_bm.l + chol_bm.u

            band_of_inv_bm = bla.band_of_inverse_from_chol(chol_bm)
            assert not np.may_share_memory(band_of_inv_bm.data, chol_bm.data)

            mat_bm = (bm.dot_mm(chol_bm, chol_bm.T) if chol_bm.u == 0
                      else bm.dot_mm(chol_bm.T, chol_bm))
            band_of_inv_full_good = fl.band_ec(
                depth, depth,
                np.eye(0, 0) if size == 0 else la.inv(mat_bm.full())
            )
            assert_allclose(band_of_inv_bm.full(), band_of_inv_full_good)

    def test_band_of_inverse(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm = gen_pos_def_BandMat(size)
            depth = mat_bm.l

            band_of_inv_bm = bla.band_of_inverse(mat_bm)
            assert not np.may_share_memory(band_of_inv_bm.data, mat_bm.data)

            band_of_inv_full_good = fl.band_ec(
                depth, depth,
                np.eye(0, 0) if size == 0 else la.inv(mat_bm.full())
            )
            assert_allclose(band_of_inv_bm.full(), band_of_inv_full_good)

if __name__ == '__main__':
    unittest.main()
