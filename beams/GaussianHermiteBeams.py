"""
File: GaussianLaguerreBeams.py
Author: Paul Grimes 
Github: https://github.com/PaulKGrimes/GaussianBeams
Description: 
"""

import numpy as np

from .GaussianHermite import gauss_herm_modes as GHM


class GaussHermiteModeBase():
    """
    The base class of the GLM and modified GLM classes.  This base class implements
    the common parameters and handles the storage and manipulation of the mode coefficients
    """

    def __init__(self, w0=1., k=1., maxL=0, maxM=0):
        self._k = k    # The wavenumber of the G-L modes
        self._w0 = w0  # The beam waist radius of the G-L modes
        self._shape = (maxL + 1, maxM + 1)

    @property
    def k(self):
        """
        return wavenumber k
        """

        return self._k

    @k.setter
    def k(self, new_k):
        """
        Set the value of k
        """

        self._k = new_k

    @property
    def w0(self):
        """
        return the beam waist radius w0
        """

        return self._w0

    @w0.setter
    def w0(self, new_w0):
        """
        Set the value of w0
        """

        self._w0 = new_w0

    @property
    def shape(self):
        """
        return the beam waist radius w0
        """

        return self._shape

class GaussHermiteModeSet(GaussHermiteModeBase):
    """
    A class holding a set of Gauss-Laguerre modes, defined in
    the paraxial limit.
    """

    def __init__(self, w0=1., k=1., maxL=0, maxM=0):
        super().__init__(w0, k, maxL, maxM)


    def field(self, x, y, z, coeffs):
        """
        Return the value of the field at r, phi, z; either for the sum of
        all modes (p, l) = None, for a specified axial mode
        p (sum over azimuthal modes), or for a specific (p, l) mode.
        """

        if self._shape != coeffs.shape:
            raise ValueError("GaussLaguerreModeSet.field: must set mode index p \
                          if mode index l is set")

        result = np.zeros_like(x, dtype=np.complex64)

        for l in range(coeffs.shape[0]):
            for m in range(coeffs.shape[1]):
                result += coeffs[l, m] * GHM(x, y, z, self.w0,
                                             self.k, l=l, m=m)
        return result
