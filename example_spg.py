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

# Copyright 2013, 2014, 2015, 2016, 2017 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import numpy as np

import bandmat as bm
import bandmat.linalg as bla

def build_win_mats(windows, frames):
    """Builds a window matrix of a given size for each window in a collection.

    `windows` specifies the collection of windows as a sequence of
    `(l, u, win_coeff)` triples, where `l` and `u` are non-negative integers
    specifying the left and right extents of the window and `win_coeff` is an
    array specifying the window coefficients.
    The returned value is a list of window matrices, one for each of the
    windows specified in `windows`.
    Each window matrix is a `frames` by `frames` Toeplitz matrix with lower
    bandwidth `l` and upper bandwidth `u`.
    The non-zero coefficients in each row of this Toeplitz matrix are given by
    `win_coeff`.
    The returned window matrices are stored as BandMats, i.e. using a banded
    representation.
    """
    win_mats = []
    for l, u, win_coeff in windows:
        assert l >= 0 and u >= 0
        assert len(win_coeff) == l + u + 1
        win_coeffs = np.tile(np.reshape(win_coeff, (l + u + 1, 1)), frames)
        win_mat = bm.band_c_bm(u, l, win_coeffs).T
        win_mats.append(win_mat)

    return win_mats

def build_poe(b_frames, tau_frames, win_mats, sdw=None):
    r"""Computes natural parameters for a Gaussian product-of-experts model.

    The natural parameters (b-value vector and precision matrix) are returned.
    The returned precision matrix is stored as a BandMat.
    Mathematically the b-value vector is given as:

        b = \sum_d \transpose{W_d} \tilde{b}_d

    and the precision matrix is given as:

        P = \sum_d \transpose{W_d} \text{diag}(\tilde{tau}_d) W_d

    where $W_d$ is the window matrix for window $d$ as specified by an element
    of `win_mats`, $\tilde{b}_d$ is the sequence over time of b-value
    parameters for window $d$ as given by a column of `b_frames`, and
    $\tilde{\tau}_d$ is the sequence over time of precision parameters for
    window $d$ as given by a column of `tau_frames`.
    """
    if sdw is None:
        sdw = max([ win_mat.l + win_mat.u for win_mat in win_mats ])
    num_windows = len(win_mats)
    frames = len(b_frames)
    assert np.shape(b_frames) == (frames, num_windows)
    assert np.shape(tau_frames) == (frames, num_windows)
    assert all([ win_mat.l + win_mat.u <= sdw for win_mat in win_mats ])

    b = np.zeros((frames,))
    prec = bm.zeros(sdw, sdw, frames)

    for win_index, win_mat in enumerate(win_mats):
        bm.dot_mv_plus_equals(win_mat.T, b_frames[:, win_index], target=b)
        bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=prec,
                              diag=tau_frames[:, win_index])

    return b, prec

def simple_example_with_random_parameters():
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]
    num_windows = len(windows)

    frames = 10
    mean_frames = np.random.randn(frames, num_windows)
    var_frames = np.abs(np.random.randn(frames, num_windows))

    b_frames = mean_frames / var_frames
    tau_frames = 1.0 / var_frames
    win_mats = build_win_mats(windows, frames)
    b, prec = build_poe(b_frames, tau_frames, win_mats)
    mean_traj = bla.solveh(prec, b)
    print 'INPUT'
    print '-----'
    print 'mean parameters over time:'
    print mean_frames
    print 'variance parameters over time:'
    print var_frames
    print
    print 'OUTPUT'
    print '------'
    print 'mean trajectory (= maximum probability trajectory):'
    print mean_traj

if __name__ == '__main__':
    simple_example_with_random_parameters()
