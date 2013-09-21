"""Tests for the bandmat package."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from bandmat.testhelp import assert_allclose, assert_allequal

import bandmat as bm

import unittest
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import random
from numpy.random import randn, randint

def randBool():
    return randint(0, 2) == 0

def trans(mat, use_transpose):
    if use_transpose:
        return mat.T
    else:
        return mat

def gen_banded_shape(l = None, u = None, size = None, max_size = None):
    if l is None:
        l = random.choice([0, 1, randint(0, 10)])
    if u is None:
        u = random.choice([0, 1, randint(0, 10)])
    if size is None:
        while True:
            size = random.choice(
                [0, 1, randint(0, 10), randint(0, 100), randint(0, 1000)]
            )
            if max_size is None or size <= max_size:
                break
    assert max_size is None or size <= max_size
    return l, u, size

def gen_banded_mat((l, u), size):
    a_banded = np.zeros((u + 1 + l, size))
    a_banded[u] = randn(size)
    for offset in range(1, u + 1):
        if offset <= size:
            a_banded[u - offset, offset:size] = randn(size - offset)
    for offset in range(1, l + 1):
        if offset <= size:
            a_banded[u + offset, 0:(size - offset)] = randn(size - offset)
    return a_banded

def gen_symmetric_banded_mat(subDiagWidth, size):
    a_banded = gen_banded_mat((subDiagWidth, subDiagWidth), size)
    a_banded += bm.transpose_banded((subDiagWidth, subDiagWidth), a_banded)
    return a_banded

def gen_pos_def_banded_mat(subDiagWidth, size, contribRank = 2):
    assert contribRank >= 0
    mat_banded = np.zeros((subDiagWidth * 2 + 1, size))
    for _ in range(contribRank):
        chol_banded = gen_banded_mat((subDiagWidth, 0), size)
        bm.dot_matrix_banded_add(
            (subDiagWidth, 0, chol_banded),
            (subDiagWidth, 0, chol_banded),
            (subDiagWidth, subDiagWidth, mat_banded),
            transpose_b = True
        )
    return mat_banded

class TestBandMat(unittest.TestCase):
    def test_bandize_and_unbandize(self, its = 100):
        for it in range(its):
            l, u, size = gen_banded_shape(max_size = 200)
            a_banded = gen_banded_mat((l, u), size)
            a = bm.unbandize((l, u), a_banded)
            a_banded_again = bm.bandize((l, u), a)
            assert_allequal(a_banded_again, a_banded)
            a_again = bm.unbandize((l, u), a_banded)
            assert_allequal(a_again, a)

    def test_zeroBandedOutsideMatrix(self, its = 100):
        for it in range(its):
            l, u, size = gen_banded_shape()
            a_banded_bad = randn(u + 1 + l, size)

            a_banded = a_banded_bad.copy()
            bm.zeroBandedOutsideMatrix((l, u), a_banded)

            a_banded_ref = bm.bandize((l, u), bm.unbandize((l, u),
                                                           a_banded_bad))

            assert_allequal(a_banded, a_banded_ref)

    def test_transpose_banded(self, its = 100):
        for it in range(its):
            l, u, size = gen_banded_shape()
            a_banded = gen_banded_mat((l, u), size)
            assert_allequal(
                bm.transpose_banded((l, u), a_banded),
                bm.bandize((u, l), bm.unbandize((l, u), a_banded).T)
            )
            assert_allequal(
                bm.transpose_banded(
                    (u, l), bm.transpose_banded((l, u), a_banded)
                ),
                a_banded
            )

    def test_symmetrize_banded(self, its = 100):
        for it in range(its):
            subDiagWidth, _, size = gen_banded_shape()
            a_banded = gen_symmetric_banded_mat(subDiagWidth, size)
            a_banded_lower = a_banded[subDiagWidth:]
            a_banded_upper = a_banded[:(subDiagWidth + 1)]
            assert_allequal(
                bm.symmetrize_banded((subDiagWidth, 0), a_banded_lower),
                a_banded
            )
            assert_allequal(
                bm.symmetrize_banded((0, subDiagWidth), a_banded_upper),
                a_banded
            )

    def test_dot_vector_banded(self, its = 100):
        for it in range(its):
            l, u, size = gen_banded_shape()
            a_banded = gen_banded_mat((l, u), size)
            b = randn(size)
            transpose = randBool()
            in_place = randBool()

            if in_place:
                c_orig = randn(size)
                c = c_orig.copy()
                bm.dot_vector_banded(
                    (l, u), a_banded,
                    b,
                    transpose = transpose,
                    target = c
                )
                c -= c_orig
            else:
                c = bm.dot_vector_banded(
                    (l, u), a_banded,
                    b,
                    transpose = transpose
                )
            c_good = np.dot(trans(bm.unbandize((l, u), a_banded), transpose),
                            b)
            assert_allclose(c, c_good)

    def test_dot_matrix_banded(self, its = 50):
        for it in range(its):
            l_a, u_a, size = gen_banded_shape(max_size = 200)
            l_b, u_b, _ = gen_banded_shape()
            a_banded = gen_banded_mat((l_a, u_a), size)
            b_banded = gen_banded_mat((l_b, u_b), size)
            diag = None if randBool() else randn(size)
            transpose_a = randBool()
            transpose_b = randBool()
            l_a_real, u_a_real = (u_a, l_a) if transpose_a else (l_a, u_a)
            l_b_real, u_b_real = (u_b, l_b) if transpose_b else (l_b, u_b)

            l_c, u_c, c_banded = bm.dot_matrix_banded(
                (l_a, u_a, a_banded),
                (l_b, u_b, b_banded),
                diag = diag,
                transpose_a = transpose_a,
                transpose_b = transpose_b
            )
            assert l_c == l_a_real + l_b_real
            assert u_c == u_a_real + u_b_real
            assert np.shape(c_banded) == (l_c + u_c + 1, size)

            c_banded_good = bm.bandize(
                (l_c, u_c),
                np.dot(
                    np.dot(
                        trans(bm.unbandize((l_a, u_a), a_banded), transpose_a),
                        np.diag(np.ones((size,)) if diag is None else diag)
                    ),
                    trans(bm.unbandize((l_b, u_b), b_banded), transpose_b)
                )
            )
            assert_allclose(c_banded, c_banded_good)

    def test_dot_matrix_banded_partial(self, its = 50):
        for it in range(its):
            l_a, u_a, size = gen_banded_shape(max_size = 200)
            l_b, u_b, _ = gen_banded_shape()
            l_c, u_c, _ = gen_banded_shape()
            a_banded = gen_banded_mat((l_a, u_a), size)
            b_banded = gen_banded_mat((l_b, u_b), size)
            diag = None if randBool() else randn(size)
            transpose_a = randBool()
            transpose_b = randBool()
            l_a_real, u_a_real = (u_a, l_a) if transpose_a else (l_a, u_a)
            l_b_real, u_b_real = (u_b, l_b) if transpose_b else (l_b, u_b)

            c_banded = bm.dot_matrix_banded_partial(
                (l_a, u_a, a_banded),
                (l_b, u_b, b_banded),
                (l_c, u_c),
                diag = diag,
                transpose_a = transpose_a,
                transpose_b = transpose_b
            )
            assert np.shape(c_banded) == (l_c + u_c + 1, size)

            c_banded_good = bm.bandize(
                (l_c, u_c),
                np.dot(
                    np.dot(
                        trans(bm.unbandize((l_a, u_a), a_banded), transpose_a),
                        np.diag(np.ones((size,)) if diag is None else diag)
                    ),
                    trans(bm.unbandize((l_b, u_b), b_banded), transpose_b)
                ),
                checkNextUpperDiagZero = False,
                checkNextLowerDiagZero = False
            )
            assert_allclose(c_banded, c_banded_good)

    def test_dot_matrix_banded_add(self, its = 50):
        for it in range(its):
            l_a, u_a, size = gen_banded_shape(max_size = 200)
            l_b, u_b, _ = gen_banded_shape()
            a_banded = gen_banded_mat((l_a, u_a), size)
            b_banded = gen_banded_mat((l_b, u_b), size)
            diag = None if randBool() else randn(size)
            transpose_a = randBool()
            transpose_b = randBool()
            l_a_real, u_a_real = (u_a, l_a) if transpose_a else (l_a, u_a)
            l_b_real, u_b_real = (u_b, l_b) if transpose_b else (l_b, u_b)

            l_c = randint(l_a_real + l_b_real + 3)
            u_c = randint(u_a_real + u_b_real + 3)
            c_banded_orig = randn(l_c + u_c + 1, size)
            c_banded = c_banded_orig.copy()
            bm.dot_matrix_banded_add(
                (l_a, u_a, a_banded),
                (l_b, u_b, b_banded),
                diag = diag,
                transpose_a = transpose_a,
                transpose_b = transpose_b,
                target_rep = (l_c, u_c, c_banded)
            )
            c_banded -= c_banded_orig

            c_banded_good = bm.bandize(
                (l_c, u_c),
                np.dot(
                    np.dot(
                        trans(bm.unbandize((l_a, u_a), a_banded), transpose_a),
                        np.diag(np.ones((size,)) if diag is None else diag)
                    ),
                    trans(bm.unbandize((l_b, u_b), b_banded), transpose_b)
                ),
                checkNextUpperDiagZero = False,
                checkNextLowerDiagZero = False
            )
            assert_allclose(c_banded, c_banded_good)

    def test_band_of_outer_add(self, its = 50):
        for it in range(its):
            l, u, size = gen_banded_shape(max_size = 200)
            a_vec = randn(size)
            b_vec = randn(size)
            mult = randn()

            mat_banded_orig = randn(l + u + 1, size)
            mat_banded = mat_banded_orig.copy()
            bm.band_of_outer_add(a_vec, b_vec, (l, u, mat_banded),
                                 mult = mult)
            mat_banded -= mat_banded_orig

            mat_banded_good = bm.bandize(
                (l, u),
                np.outer(a_vec, b_vec) * mult,
                checkNextUpperDiagZero = False,
                checkNextLowerDiagZero = False
            )

            assert_allclose(mat_banded, mat_banded_good)

    def test_band_of_inverse_from_chol(self, its = 50):
        for it in range(its):
            depth = random.choice([0, 1, randint(0, 10)])
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            # generate random pos def matrix then compute Cholesky factor
            #   (generating random Cholesky factor directly seems to often lead
            #   to ill-conditioned matrices)
            chol_banded = sla.cholesky_banded(
                gen_pos_def_banded_mat(depth, size)[depth:],
                lower = True
            )

            cov_banded = bm.band_of_inverse_from_chol(chol_banded)

            _, _, prec_banded = bm.dot_matrix_banded(
                (depth, 0, chol_banded),
                (depth, 0, chol_banded),
                transpose_b = True
            )
            cov_banded_good = bm.bandize(
                (depth, depth),
                np.eye(0, 0) if size == 0 else la.inv(
                    bm.unbandize((depth, depth), prec_banded)
                ),
                checkNextUpperDiagZero = False,
                checkNextLowerDiagZero = False
            )

            assert_allclose(cov_banded, cov_banded_good)

    def test_band_of_inverse(self, its = 50):
        for it in range(its):
            depth = random.choice([0, 1, randint(0, 10)])
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            prec_banded = gen_pos_def_banded_mat(depth, size)

            cov_banded = bm.band_of_inverse(prec_banded)

            cov_banded_good = bm.bandize(
                (depth, depth),
                np.eye(0, 0) if size == 0 else la.inv(
                    bm.unbandize((depth, depth), prec_banded)
                ),
                checkNextUpperDiagZero = False,
                checkNextLowerDiagZero = False
            )

            assert_allclose(cov_banded, cov_banded_good)

    def test_plusEquals(self, its = 100):
        for it in range(its):
            sourceSize = random.choice([0, 1, randint(10), randint(100)])
            targetSize = random.choice([1, randint(1, 10), randint(1, 100)])
            source = randn(sourceSize)
            target = randn(targetSize)
            targetIndexSeq = np.array([ randint(targetSize)
                                        for _ in range(sourceSize) ],
                                      dtype = np.int64)

            targetGood = target.copy()
            for sourceIndex, targetIndex in enumerate(targetIndexSeq):
                targetGood[targetIndex] += source[sourceIndex]

            bm.plusEquals(targetIndexSeq, source, target)

            assert_allclose(target, targetGood)

    def test_plusEquals_fails(self, its = 100):
        for it in range(its):
            sourceSize = random.choice([1, randint(1, 10), randint(1, 100)])
            targetSize = random.choice([1, randint(1, 10), randint(1, 100)])
            source = randn(sourceSize)
            target = randn(targetSize)
            targetIndexSeq = np.array([ randint(targetSize)
                                        for _ in range(sourceSize) ],
                                      dtype = np.int64)
            targetIndexSeq[randint(sourceSize)] = (
                (-targetSize - 1 - randint(10)) if randBool()
                else targetSize + randint(10)
            )
            self.assertRaises(IndexError,
                              bm.plusEquals,
                              targetIndexSeq, source, target)

    def test_plusEquals2D(self, its = 100):
        for it in range(its):
            sourceSize = random.choice([0, 1, randint(10), randint(100)])
            targetSize = random.choice([1, randint(1, 10), randint(1, 100)])
            size1 = random.choice([0, 1, randint(10)])
            source = randn(sourceSize, size1)
            target = randn(targetSize, size1)
            targetIndexSeq = np.array([ randint(targetSize)
                                        for _ in range(sourceSize) ],
                                      dtype = np.int64)

            targetGood = target.copy()
            for sourceIndex, targetIndex in enumerate(targetIndexSeq):
                targetGood[targetIndex] += source[sourceIndex]

            bm.plusEquals2D(targetIndexSeq, source, target)

            assert_allclose(target, targetGood)

    def test_plusEquals2D_fails(self, its = 100):
        for it in range(its):
            sourceSize = random.choice([1, randint(1, 10), randint(1, 100)])
            targetSize = random.choice([1, randint(1, 10), randint(1, 100)])
            size1 = random.choice([1, randint(1, 10)])
            source = randn(sourceSize, size1)
            target = randn(targetSize, size1)
            targetIndexSeq = np.array([ randint(targetSize)
                                        for _ in range(sourceSize) ],
                                      dtype = np.int64)
            targetIndexSeq[randint(sourceSize)] = (
                (-targetSize - 1 - randint(10)) if randBool()
                else targetSize + randint(10)
            )
            self.assertRaises(IndexError,
                              bm.plusEquals2D,
                              targetIndexSeq, source, target)

if __name__ == '__main__':
    unittest.main()
