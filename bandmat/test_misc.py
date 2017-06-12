"""Tests for assorted helpful functions."""

# Copyright 2013, 2014, 2015, 2016, 2017 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import unittest
import random
from numpy.random import randn, randint

from bandmat.misc import fancy_plus_equals, fancy_plus_equals_2d
from bandmat.misc import fancy_plus_equals_3d
from bandmat.testhelp import assert_allclose, get_array_mem

def rand_bool():
    return randint(0, 2) == 0

class TestMisc(unittest.TestCase):
    def test_fancy_plus_equals(self, its=100):
        for it in range(its):
            source_size = random.choice([0, 1, randint(10), randint(100)])
            target_size = random.choice([1, randint(1, 10), randint(1, 100)])
            source = randn(source_size)
            target = randn(target_size)
            target_index_seq = randint(target_size, size=source_size)
            array_mem = get_array_mem(target_index_seq, source, target)

            target_good = target.copy()
            for source_index, target_index in enumerate(target_index_seq):
                target_good[target_index] += source[source_index]

            fancy_plus_equals(target_index_seq, source, target)

            assert_allclose(target, target_good)
            assert get_array_mem(target_index_seq, source, target) == array_mem

    def test_fancy_plus_equals_fails(self, its=100):
        for it in range(its):
            source_size = random.choice([1, randint(1, 10), randint(1, 100)])
            target_size = random.choice([1, randint(1, 10), randint(1, 100)])
            source = randn(source_size)
            target = randn(target_size)
            target_index_seq = randint(target_size, size=source_size)
            target_index_seq[randint(source_size)] = (
                (-target_size - 1 - randint(10)) if rand_bool()
                else target_size + randint(10)
            )
            self.assertRaises(IndexError,
                              fancy_plus_equals,
                              target_index_seq, source, target)

    def test_fancy_plus_equals_2d(self, its=100):
        for it in range(its):
            source_size = random.choice([0, 1, randint(10), randint(100)])
            target_size = random.choice([1, randint(1, 10), randint(1, 100)])
            size1 = random.choice([0, 1, randint(10)])
            source = randn(source_size, size1)
            target = randn(target_size, size1)
            target_index_seq = randint(target_size, size=source_size)
            array_mem = get_array_mem(target_index_seq, source, target)

            target_good = target.copy()
            for source_index, target_index in enumerate(target_index_seq):
                target_good[target_index] += source[source_index]

            fancy_plus_equals_2d(target_index_seq, source, target)

            assert_allclose(target, target_good)
            assert get_array_mem(target_index_seq, source, target) == array_mem

    def test_fancy_plus_equals_2d_fails(self, its=100):
        for it in range(its):
            source_size = random.choice([1, randint(1, 10), randint(1, 100)])
            target_size = random.choice([1, randint(1, 10), randint(1, 100)])
            size1 = random.choice([1, randint(1, 10)])
            source = randn(source_size, size1)
            target = randn(target_size, size1)
            target_index_seq = randint(target_size, size=source_size)
            target_index_seq[randint(source_size)] = (
                (-target_size - 1 - randint(10)) if rand_bool()
                else target_size + randint(10)
            )
            self.assertRaises(IndexError,
                              fancy_plus_equals_2d,
                              target_index_seq, source, target)

    def test_fancy_plus_equals_3d(self, its=100):
        for it in range(its):
            source_size = random.choice([0, 1, randint(10), randint(100)])
            target_size = random.choice([1, randint(1, 10), randint(1, 100)])
            size1 = random.choice([0, 1, randint(10)])
            size2 = random.choice([0, 1, randint(10)])
            source = randn(source_size, size1, size2)
            target = randn(target_size, size1, size2)
            target_index_seq = randint(target_size, size=source_size)
            array_mem = get_array_mem(target_index_seq, source, target)

            target_good = target.copy()
            for source_index, target_index in enumerate(target_index_seq):
                target_good[target_index] += source[source_index]

            fancy_plus_equals_3d(target_index_seq, source, target)

            assert_allclose(target, target_good)
            assert get_array_mem(target_index_seq, source, target) == array_mem

    def test_fancy_plus_equals_3d_fails(self, its=100):
        for it in range(its):
            source_size = random.choice([1, randint(1, 10), randint(1, 100)])
            target_size = random.choice([1, randint(1, 10), randint(1, 100)])
            size1 = random.choice([1, randint(1, 10)])
            size2 = random.choice([1, randint(1, 10)])
            source = randn(source_size, size1, size2)
            target = randn(target_size, size1, size2)
            target_index_seq = randint(target_size, size=source_size)
            target_index_seq[randint(source_size)] = (
                (-target_size - 1 - randint(10)) if rand_bool()
                else target_size + randint(10)
            )
            self.assertRaises(IndexError,
                              fancy_plus_equals_3d,
                              target_index_seq, source, target)

if __name__ == '__main__':
    unittest.main()
