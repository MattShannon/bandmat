"""Tests that the example scripts run successfully."""

# Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import unittest
import subprocess
import sys
import os

example_dir = os.path.join(os.path.dirname(__file__), '..')

class TestExamples(unittest.TestCase):
    def test_example(self):
        """Checks example.py runs without error (has its own assertions)."""
        subprocess.check_call(
            [sys.executable, os.path.join(example_dir, 'example.py')]
        )

    def test_example_spg(self):
        """Checks example_spg.py runs without error."""
        subprocess.check_call(
            [sys.executable, os.path.join(example_dir, 'example_spg.py')]
        )

if __name__ == '__main__':
    unittest.main()
