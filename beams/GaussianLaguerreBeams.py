"""
File: GaussianLaguerreBeams.py
Author: Paul Grimes 
Github: https://github.com/PaulKGrimes/GaussianBeams
Description: 
"""

import numpy as np

from .GaussianLaguerre import gauss_lag_modes as GLM
from .ModifiedGaussianLaguerre import mod_gauss_lag_modes_ff as modGLM_ff
from .ModifiedGaussianLaguerre import mod_gauss_lag_modes_nf as modGLM_nf


class GaussLaguerreModeBase():
    """
    The base class of the GLM and modified GLM classes.  This base class implements
    the common parameters and handles the storage and manipulation of the mode coefficients
    """

    def __init__(self, w0=1., k=1., maxP=0, maxL=0):
        self._k = k    # The wavenumber of the G-L modes
        self._w0 = w0  # The beam waist radius of the G-L modes
        self._shape = (maxP + 1, 2*maxL + 1)

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

class GaussLaguerreModeSet(GaussLaguerreModeBase):
    """
    A class holding a set of Gauss-Laguerre modes, defined in
    the paraxial limit.
    """

    def __init__(self, w0=1., k=1., maxP=0, maxL=0):
        super().__init__(w0, k, maxP, maxL)


    def field(self, r, phi, z, coeffs):
        """
        Return the value of the field at r, phi, z; either for the sum of
        all modes (p, l) = None, for a specified axial mode
        p (sum over azimuthal modes), or for a specific (p, l) mode.
        """

        if self._shape != coeffs.shape:
            raise ValueError("GaussLaguerreModeSet.field: must set mode index p \
                          if mode index l is set")

        result = np.zeros_like(r, dtype=np.complex64)

        for p in range(coeffs.shape[0]):
            for l in range(coeffs.shape[1]):
                result += coeffs[p, l] * GLM(r, phi, z, self.w0,
                                             self.k, p=p, l=l)
        return result


class ModifiedGaussLaguerreModeSet(GaussLaguerreModeBase):
    """
    A class holding a set of modified Gauss-Laguerre modes,
    using the definition Tuovinen (1992)
    """

    def __init__(self, w0=1., k=1., maxP=0, maxL=0):
        super(ModifiedGaussLaguerreModeSet, self).__init__(w0, k, maxP, maxL)

    def near_field(self, r, phi, z, coeffs):
        """
        Return the value of the field at r, phi, z; either for the sum of all
        modes (p, l) = None, for a specified axial mode 
        p (sum over azimuthal modes), or for a specific (p, l) mode.
        """


        if self._shape != coeffs.shape:
            raise ValueError("GaussLaguerreModeSet.field: must set mode index p \
                          if mode index l is set")

        result = np.zeros_like(r, dtype=np.complex64)
        for p in range(coeffs.shape[0]):
            for l in range(coeffs.shape[1]):
                result += coeffs[p, l] * modGLM_nf(r, phi, z, self.w0,
                                                   self.k, p=p, l=l)
        return result

    def far_field(self, theta, phi, z, coeffs):
        """
        Return the value of the far field at theta, phi; either for the sum of all
        modes (p, l) = None, for a specified axial mode
        p (sum over azimuthal modes), or for a specific (p, l) mode.
        """

        if coeffs.shape != self._shape:
            raise ValueError("GaussLaguerreModeSet.field: must set mode index p \
                          if mode index l is set")

        result = np.zeros_like(theta, dtype=np.complex64)
        for p in range(coeffs.shape[0]):
            for l in range(coeffs.shape[1]):
                result += coeffs[p, l] * modGLM_ff(theta, phi, z, self.w0,
                                                   self.k, p=p, l=l)
        return result
