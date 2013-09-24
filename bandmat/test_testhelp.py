"""Tests for helper functions for testing."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import bandmat as bm
import bandmat.full as fl
import bandmat.testhelp as th

import unittest
import numpy as np
import random
from numpy.random import randn, randint

def randBool():
    return randint(0, 2) == 0

def gen_BandMat_simple(size):
    """Generates a random BandMat."""
    l = random.choice([0, 1, randint(0, 10)])
    u = random.choice([0, 1, randint(0, 10)])
    data = randn(l + u + 1, size)
    transposed = randBool()
    return bm.BandMat(l, u, data, transposed = transposed)

class TestTestHelp(unittest.TestCase):
    def test_assert_allclose(self):
        a0 = np.array([2.0, 3.0, 4.0])
        a1 = np.array([2.0, 3.0, 4.0])
        a2 = np.array([2.0, 3.0])
        a3 = np.array([2.0, 3.0, 5.0])
        a4 = np.array([[2.0, 3.0, 4.0]])
        th.assert_allclose(a0, a0)
        th.assert_allclose(a0, a1)
        self.assertRaises(AssertionError, th.assert_allclose, a0, a2)
        self.assertRaises(AssertionError, th.assert_allclose, a0, a3)
        self.assertRaises(AssertionError, th.assert_allclose, a0, a4)

    def test_assert_allequal(self):
        a0 = np.array([2.0, 3.0, 4.0])
        a1 = np.array([2.0, 3.0, 4.0])
        a2 = np.array([2.0, 3.0])
        a3 = np.array([2.0, 3.0, 5.0])
        a4 = np.array([[2.0, 3.0, 4.0]])
        th.assert_allequal(a0, a0)
        th.assert_allequal(a0, a1)
        self.assertRaises(AssertionError, th.assert_allequal, a0, a2)
        self.assertRaises(AssertionError, th.assert_allequal, a0, a3)
        self.assertRaises(AssertionError, th.assert_allequal, a0, a4)

    def test_randomize_extra_entries(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            l = random.choice([0, 1, randint(0, 10)])
            u = random.choice([0, 1, randint(0, 10)])

            mat_rect = randn(l + u + 1, size)
            assert np.all(mat_rect != 0.0)
            fl.zero_extra_entries(l, u, mat_rect)
            th.randomize_extra_entries(l, u, mat_rect)
            assert np.all(mat_rect != 0.0)

            mat_rect = np.zeros((l + u + 1, size))
            assert np.all(mat_rect == 0.0)
            th.randomize_extra_entries(l, u, mat_rect)
            fl.zero_extra_entries(l, u, mat_rect)
            assert np.all(mat_rect == 0.0)

    def test_randomize_extra_entries_bm(self, its = 50):
        for it in range(its):
            size = random.choice([0, 1, randint(0, 10), randint(0, 100)])
            mat_bm = gen_BandMat_simple(size)

            mat_full = mat_bm.full()
            th.randomize_extra_entries_bm(mat_bm)
            th.assert_allequal(mat_bm.full(), mat_full)

if __name__ == '__main__':
    unittest.main()
