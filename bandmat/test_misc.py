"""Tests for assorted helpful functions."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

from bandmat.testhelp import assert_allclose, assert_allequal, get_array_mem

from bandmat.misc import fancy_plus_equals, fancy_plus_equals_2d

import unittest
import numpy as np
import random
from numpy.random import randn, randint

def randBool():
    return randint(0, 2) == 0

class TestMisc(unittest.TestCase):
    def test_fancy_plus_equals(self, its = 100):
        for it in range(its):
            sourceSize = random.choice([0, 1, randint(10), randint(100)])
            targetSize = random.choice([1, randint(1, 10), randint(1, 100)])
            source = randn(sourceSize)
            target = randn(targetSize)
            targetIndexSeq = np.array([ randint(targetSize)
                                        for _ in range(sourceSize) ],
                                      dtype = np.int64)
            array_mem = get_array_mem(targetIndexSeq, source, target)

            targetGood = target.copy()
            for sourceIndex, targetIndex in enumerate(targetIndexSeq):
                targetGood[targetIndex] += source[sourceIndex]

            fancy_plus_equals(targetIndexSeq, source, target)

            assert_allclose(target, targetGood)
            assert get_array_mem(targetIndexSeq, source, target) == array_mem

    def test_fancy_plus_equals_fails(self, its = 100):
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
                              fancy_plus_equals,
                              targetIndexSeq, source, target)

    def test_fancy_plus_equals_2d(self, its = 100):
        for it in range(its):
            sourceSize = random.choice([0, 1, randint(10), randint(100)])
            targetSize = random.choice([1, randint(1, 10), randint(1, 100)])
            size1 = random.choice([0, 1, randint(10)])
            source = randn(sourceSize, size1)
            target = randn(targetSize, size1)
            targetIndexSeq = np.array([ randint(targetSize)
                                        for _ in range(sourceSize) ],
                                      dtype = np.int64)
            array_mem = get_array_mem(targetIndexSeq, source, target)

            targetGood = target.copy()
            for sourceIndex, targetIndex in enumerate(targetIndexSeq):
                targetGood[targetIndex] += source[sourceIndex]

            fancy_plus_equals_2d(targetIndexSeq, source, target)

            assert_allclose(target, targetGood)
            assert get_array_mem(targetIndexSeq, source, target) == array_mem

    def test_fancy_plus_equals_2d_fails(self, its = 100):
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
                              fancy_plus_equals_2d,
                              targetIndexSeq, source, target)

if __name__ == '__main__':
    unittest.main()
