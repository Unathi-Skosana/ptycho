"""
File: GaussianHermiteBeams.py
Author: Unathi Skosana
Github: https://github.com/UnathiSkosana/ptycho
Description: 

Adapted from https://github.com/PaulKGrimes/GaussianBeams
"""

import numpy as np

from .GaussianHermite import gauss_herm_modes as GHM


class GaussHermiteModeBase():
    """
    The base class of the GHM and modified GHM classes.
    """

    def __init__(self, w0=1., k=1., maxL=0, maxM=0):
        """ Initialize class instance """
        self._k = k    # The wavenumber of the G-L modes
        self._w0 = w0  # The beam waist radius of the G-L modes
        self._shape = (maxL + 1, maxM + 1)

    @property
    def k(self):
        """ Get wavenumber k """

        return self._k

    @k.setter
    def k(self, new_k):
        """ Set the value of k """

        self._k = new_k

    @property
    def w0(self):
        """ Get beam waist radius w0 """

        return self._w0

    @w0.setter
    def w0(self, new_w0):
        """ Set the value of w0 """

        self._w0 = new_w0

    @property
    def shape(self):
        """ Get shape of mode coefficients shape """

        return self._shape


class GaussHermiteModeSet(GaussHermiteModeBase):
    """
    A class holding a set of Gauss-Hermite modes, defined in
    the paraxial limit.
    """

    def __init__(self, w0=1., k=1., maxL=0, maxM=0):
        """ Initialize class instance """
        super().__init__(w0, k, maxL, maxM)

    def field(self, x, y, z, coeffs):
        """
        Computes the value of the field at x, y, z for a sum of
        all modes in coeffs

        Args:
            x: distance from the center axis of the beam in the x-direction
            y: distance from the center axis of the beam in the y-direction
            z: axial distance from the beam waist
            coeffs:  coefficients of all (maxL + 1)(maxM + 1) modes
        """

        if self._shape != coeffs.shape:
            raise ValueError("GaussHermiteModeSet.field: coefficients must \
                        have a shape of {}".format(self._shape))

        result = np.zeros_like(x, dtype=np.complex64)

        for l in range(coeffs.shape[0]):
            for m in range(coeffs.shape[1]):
                result += coeffs[l, m] * GHM(x, y, z, self.w0,
                                             self.k, l=l, m=m)
        return result
