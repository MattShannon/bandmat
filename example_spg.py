#!/usr/bin/python
"""An example of using the bandmat package for a non-trivial application.

This script demonstrates how the bandmat package can be used to write a simple
but very efficient implementation of the maximum probability speech parameter
generation algorithm used in statistical speech synthesis.
(This algorithm is also sometimes known as maximum likelihood speech parameter
generation, but this is a misnomer since the quantity being maximized is not a
function of the model parameters and is therefore not a likelihood.)
Given a sequence of mean and variance parameters over time, the maximum
probability speech parameter generation algorithm computes the natural
parameters of a corresponding Gaussian distribution over trajectories, and then
computes the mean trajectory using a banded Cholesky decomposition.
The mean trajectory is also the most likely or maximum probability trajectory.
"""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import bandmat as bm
import bandmat.linalg as bla

import numpy as np

def build_win_mats(windows, frames):
    """Builds a window matrix of a given size for each window in a collection.

    `windows` specifies the collection of windows as a sequence of
    `(l, u, winCoeff)` triples, where `l` and `u` are non-negative integers
    specifying the left and right extents of the window and `winCoeff` is an
    array specifying the window coefficients.
    The returned value is a list of window matrices, one for each of the
    windows specified in `windows`.
    Each window matrix is a `frames` by `frames` Toeplitz matrix with lower
    bandwidth `l` and upper bandwidth `u`.
    The non-zero coefficients in each row of this Toeplitz matrix are given by
    `winCoeff`.
    The returned window matrices are stored as BandMats, i.e. using a banded
    representation.
    """
    winMats = []
    for l, u, winCoeff in windows:
        assert l >= 0 and u >= 0
        assert len(winCoeff) == l + u + 1
        winCoeffs = np.tile(np.reshape(winCoeff, (l + u + 1, 1)), frames)
        winMat = bm.band_c_bm(u, l, winCoeffs).T
        winMats.append(winMat)

    return winMats

def build_poe(bFrames, tauFrames, winMats, sdw = None):
    r"""Computes natural parameters for a Gaussian product-of-experts model.

    The natural parameters (b-value vector and precision matrix) are returned.
    The returned precision matrix is stored as a BandMat.
    Mathematically the b-value vector is given as:

        b = \sum_d \transpose{W_d} \tilde{b}_d

    and the precision matrix is given as:

        P = \sum_d \transpose{W_d} \text{diag}(\tilde{tau}_d) W_d

    where $W_d$ is the window matrix for window $d$ as specified by an element
    of `winMats`, $\tilde{b}_d$ is the sequence over time of b-value parameters
    for window $d$ as given by a column of `bFrames`, and $\tilde{\tau}_d$ is
    the sequence over time of precision parameters for window $d$ as given by
    a column of `tauFrames`.
    """
    if sdw is None:
        sdw = max([ winMat.l + winMat.u for winMat in winMats ])
    numWindows = len(winMats)
    frames = len(bFrames)
    assert np.shape(bFrames) == (frames, numWindows)
    assert np.shape(tauFrames) == (frames, numWindows)
    assert all([ winMat.l + winMat.u <= sdw for winMat in winMats ])

    b = np.zeros((frames,))
    prec = bm.zeros(sdw, sdw, frames)

    for winIndex, winMat in enumerate(winMats):
        bm.dot_mv_plus_equals(winMat.T, bFrames[:, winIndex], target = b)
        bm.dot_mm_plus_equals(winMat.T, winMat, target_bm = prec,
                              diag = tauFrames[:, winIndex])

    return b, prec

def simple_example_with_random_parameters():
    windows = [
        (0, 0, np.array([0.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]
    numWindows = len(windows)

    frames = 10
    meanFrames = np.random.randn(frames, numWindows)
    varFrames = np.abs(np.random.randn(frames, numWindows))

    bFrames = meanFrames / varFrames
    tauFrames = 1.0 / varFrames
    winMats = build_win_mats(windows, frames)
    b, prec = build_poe(bFrames, tauFrames, winMats)
    meanTraj = bla.solveh(prec, b)
    print 'INPUT'
    print '-----'
    print 'mean parameters over time:'
    print meanFrames
    print 'variance parameters over time:'
    print varFrames
    print
    print 'OUTPUT'
    print '------'
    print 'mean trajectory (= maximum probability trajectory):'
    print meanTraj

if __name__ == '__main__':
    simple_example_with_random_parameters()
