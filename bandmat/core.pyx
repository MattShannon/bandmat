"""Core banded matrix definitions and functions."""

# Copyright 2013 Matt Shannon

# This file is part of bandmat.
# See `License` for details of license and warranty.

import bandmat.full as fl

import numpy as np

cimport numpy as cnp
cimport cython

cnp.import_array()
cnp.import_ufunc()

class BandMat(object):
    """A memory-efficient representation of a square banded matrix.

    An N by N matrix with bandwidth D can be stored efficiently by storing its
    band as a rectangular D by N matrix.
    This class is a lightweight wrapper around a rectangular matrix being used
    in this way, and stores additional details needed to recover the square
    matrix such as the lower and upper bandwidths.

    The representation used for the rectangular matrix is the same one used by
    BLAS and LAPACK (and thus scipy): the columns of the rectangular matrix are
    (parts of) the columns of the square matrix being represented, and the
    successive rows of the rectangular matrix give the superdiagonals, diagonal
    and subdiagonals in order (starting with the outermost superdiagonal and
    ending with the outermost subdiagonal).
    See the "Band Storage" section of the LAPACK Users' Guide at
    http://www.netlib.org/lapack/lug/node124.html or the docstring for
    scipy.linalg.solve_banded for some examples.

    `l` is the number of subdiagonals stored and `u` is the number of
    superdiagonals stored.
    Thus `l` and `u` determine the band in which the entries of the represented
    matrix can be non-zero.
    `data` is the LAPACK-style banded matrix representation of the square
    matrix (or of the transpose of the square matrix if `transposed` is True)
    stored as a numpy array.
    Note that if `transposed` is True, `l` and `u` still refer to the square
    matrix being represented rather than to its transpose.
    """
    def __init__(self, l, u, data, transposed = False):
        self.l = l
        self.u = u
        self.data = data
        self.transposed = transposed

        assert self.l >= 0
        assert self.u >= 0
        assert self.data.ndim == 2
        assert self.data.shape[0] == self.l + self.u + 1

    def __repr__(self):
        return ('BandMat(%r, %r, %r, transposed=%r)' %
                (self.l, self.u, self.data, self.transposed))

    @property
    def size(self):
        """Returns the size of this matrix."""
        return self.data.shape[1]

    @property
    def T(self):
        """Returns the transpose of this matrix.

        This is a cheap operation since it just sets a flag internally.
        The returned BandMat has the same underlying data array as `self`.
        """
        return BandMat(self.u, self.l, self.data,
                       transposed = not self.transposed)

    def full(self):
        """Converts this BandMat to a conventional numpy array.

        The returned numpy array represents the same matrix as `self`.
        """
        if self.transposed:
            return fl.band_c(self.u, self.l, self.data).T
        else:
            return fl.band_c(self.l, self.u, self.data)

    def copy_exact(self):
        """Returns a copy of this BandMat.

        The returned BandMat represents the same matrix as `self`, but has a
        newly-created underlying data array.
        It has the same `transposed` setting as `self`.
        """
        return BandMat(self.l, self.u, self.data.copy(),
                       transposed = self.transposed)

    def copy(self):
        """Returns a copy of this BandMat with transposed set to False.

        The returned BandMat represents the same matrix as `self`, but has a
        newly-created underlying data array, and always has `transposed` set to
        False.
        """
        l = self.l
        u = self.u
        if self.transposed:
            return BandMat(l, u, fl.band_cTe(u, l, self.data))
        else:
            return BandMat(l, u, self.data.copy())

    def equiv(self, l_new = None, u_new = None, transposed_new = None,
              zero_extra = False):
        """Returns an equivalent BandMat stored differently.

        The returned BandMat represents the same matrix as `self`, but has a
        newly-created underlying data array, and has possibly different
        parameters `l`, `u` and `transposed`.
        The new values of these parameters are given by `l_new`, `u_new` and
        `transposed_new`, with the corresponding value from `self` used if any
        of these are None.

        If `zero_extra` is True then the underlying data array of the returned
        BandMat is guaranteed to have extra entries set to zero.
        """
        l = self.l
        u = self.u
        if l_new is None:
            l_new = l
        if u_new is None:
            u_new = u
        if transposed_new is None:
            transposed_new = self.transposed

        assert l_new >= l
        assert u_new >= u

        data_new = np.empty((l_new + u_new + 1, self.size))

        ll, uu = (u, l) if transposed_new else (l, u)
        ll_new, uu_new = (u_new, l_new) if transposed_new else (l_new, u_new)

        data_new[(uu_new - uu_new):(uu_new - uu)] = 0.0
        data_new[(uu_new + ll + 1):(uu_new + ll_new + 1)] = 0.0
        data_new_co = data_new[(uu_new - uu):(uu_new + ll + 1)]
        if self.transposed == transposed_new:
            data_new_co[:] = self.data
            if zero_extra:
                fl.zero_extra_entries(ll, uu, data_new_co)
        else:
            fl.band_cTe(uu, ll, self.data, target_rect = data_new_co)

        return BandMat(l_new, u_new, data_new, transposed = transposed_new)

def zeros(l, u, size):
    """Returns the zero matrix as a BandMat.

    The returned BandMat `ret_bm` has `ret_bm.l = l`, `ret_bm.u` = `u` and
    `ret_bm.size = size`.
    """
    data = np.zeros((l + u + 1, size))
    return BandMat(l, u, data)

def from_full(l, u, mat_full):
    """Converts a square banded numpy array to a BandMat.

    The returned BandMat represents the same matrix as `mat_full`.
    `mat_full` should be a numpy array representing a square matrix with zeros
    outside the band specified by `l` and `u`.
    An AssertionError is raised if `mat_full` has non-zero entries outside the
    specified band.
    """
    mat_bm = BandMat(l, u, fl.band_e(l, u, mat_full))
    # check `mat_full` is zero outside the specified band
    assert np.all(mat_bm.full() == mat_full)
    return mat_bm

def band_c_bm(l, u, mat_rect):
    """Constructs a BandMat from its band.

    The expression `band_c_bm(l, u, mat_rect)` where `mat_rect` is a
    rectangular numpy array is the equivalent of:

        band_c(l, u, mat_rect)

    where the returned value is a square numpy array.
    """
    return BandMat(l, u, mat_rect)

def band_e_bm(l, u, mat_bm):
    """Extracts a band of a BandMat.

    The band to extract is specified by `l` and `u`.

    The expression `band_e_bm(l, u, mat_bm)` where `mat_bm` is a BandMat is the
    equivalent of:

        band_e(l, u, mat_full)

    where `mat_full` is a square numpy array.
    """
    mat_bm_co = band_ec_bm_view(l, u, mat_bm)
    mat_bm_new = mat_bm_co.equiv(l_new = l, u_new = u,
                                 transposed_new = False,
                                 zero_extra = True)
    return mat_bm_new.data

band_ce_bm = fl.band_ce

def band_ec_bm_view(l, u, mat_bm):
    """Effectively applies `band_e_bm` then `band_c_bm`, sharing data arrays.

    The combined operation has the effect of zeroing the entries outside the
    band specified by `l` and `u`.
    This is implemented by taking a view of `mat_bm`'s underlying data array.
    To obtain a BandMat with a fresh underlying data array, `.copy_exact()`
    should be called on the result.
    """
    assert l >= 0
    assert u >= 0

    l_in = mat_bm.l
    u_in = mat_bm.u
    l_out = min(l, l_in)
    u_out = min(u, u_in)
    if mat_bm.transposed:
        return BandMat(
            l_out, u_out,
            mat_bm.data[(l_in - l_out):(l_in + u_out + 1)],
            transposed = True
        )
    else:
        return BandMat(
            l_out, u_out,
            mat_bm.data[(u_in - u_out):(u_in + l_out + 1)]
        )

def band_ec_bm(l, u, mat_bm):
    """Effectively applies `band_e_bm` then `band_c_bm`.

    The combined operation has the effect of zeroing the entries outside the
    band specified by `l` and `u`.

    The expression `band_ec_bm(l, u, mat_bm)` where `mat_bm` is a BandMat is
    the equivalent of:

        band_ec(l, u, mat_full)

    where `mat_full` and the returned value are square numpy arrays.
    """
    return band_ec_bm_view(l, u, mat_bm).copy_exact()
