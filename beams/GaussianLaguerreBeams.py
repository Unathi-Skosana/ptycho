"""
File: GaussianLaguerreBeams.py
Author: Paul Grimes 
Github: https://github.com/PaulKGrimes/GaussianBeams
Description: 

Adapted from https://github.com/PaulKGrimes/GaussianBeams
"""

import numpy as np

from .GaussianLaguerre import gauss_lag_modes as GLM
from .ModifiedGaussianLaguerre import mod_gauss_lag_modes_ff as modGLM_ff
from .ModifiedGaussianLaguerre import mod_gauss_lag_modes_nf as modGLM_nf


class GaussLaguerreModeBase():
    """
    The base class of the GLM and modified GLM classes.
    """

    def __init__(self, w0=1., k=1., maxP=0, maxL=0):
        """ Initialize class instance """
        self._k = k    # The wavenumber of the G-L modes
        self._w0 = w0  # The beam waist radius of the G-L modes
        self._shape = (maxP + 1, 2*maxL + 1)

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


class GaussLaguerreModeSet(GaussLaguerreModeBase):
    """
    A class holding a set of Gauss-Laguerre modes, defined in
    the cylindrical coordinates.
    """

    def __init__(self, w0=1., k=1., maxP=0, maxL=0):
        """ Initialize class instance """
        super().__init__(w0, k, maxP, maxL)

    def field(self, r, phi, z, coeffs):
        """
        Computes the value of the field at r, phi, z for a sum of
        all modes in coeffs

        Args:
            r: radial distance from center axis of the beam
            phi: azimuthal angle between the reference direction and projection
                 of r onto the plane
            z: axial distance from the beam waist
            coeffs:  coefficients of all (maxL + 1)(2*maxP + 1) modes
        Returns:
            value of field
        Raises:
            ValueError: if the coefficients array is misshaped.
        """

        if self._shape != coeffs.shape:
            raise ValueError("GaussLaguerreModeSet.field: coefficients must \
                        have a shape of {}".format(self._shape))

        result = np.zeros_like(r, dtype=np.complex64)

        for p in range(coeffs.shape[0]):
            for l in range(coeffs.shape[1]):
                result += coeffs[p, l] * GLM(r, phi, z, self.k,
                                             self.w0, p=p, l=l)
        return result


class ModifiedGaussLaguerreModeSet(GaussLaguerreModeBase):
    """
    A class holding a set of modified Gauss-Laguerre modes,
    using the definition Tuovinen (1992)
    """

    def __init__(self, w0=1., k=1., maxP=0, maxL=0):
        super().__init__(w0, k, maxP, maxL)

    def near_field(self, r, phi, z, coeffs):
        """
        Computes the value of the near field at r, phi, z for a sum of
        all modes in coeffs

        Args:
            r: radial distance from center axis of the beam
            phi: azimuthal angle between the reference direction and projection
                 of r onto the plane
            z: axial distance from the beam waist
            coeffs:  coefficients of all (maxL + 1)(2*maxP + 1) modes
        Returns:
            value of near field
        Raises:
            ValueError: if the coefficients array is misshaped.
        """

        if self._shape != coeffs.shape:
            raise ValueError("GaussLaguerreModeSet.field: coefficients must \
                        have a shape of {}".format(self._shape))

        result = np.zeros_like(r, dtype=np.complex64)
        for p in range(coeffs.shape[0]):
            for l in range(coeffs.shape[1]):
                result += coeffs[p, l] * modGLM_nf(r, phi, z, self.k,
                                                   self.w0, p=p, l=l)
        return result

    def far_field(self, theta, phi, z, coeffs):
        """
        Computes the value of the far field at theta , phi, z for a sum of
        all modes in coeffs

        Args:
            theta: angle that parameterizes the far field i.e angle of boresight
            phi: azimuthal angle between the reference direction and projection
                 of r onto the plane
            z: axial distance from the beam waist
            coeffs:  coefficients of all (maxL + 1)(2*maxP + 1) modes
        Returns:
            value of near field
        Raises:
            ValueError: if the coefficients array is misshaped.
        """

        if self._shape != coeffs.shape:
            raise ValueError("GaussLaguerreModeSet.field: coefficients must \
                        have a shape of {}".format(self._shape))

        result = np.zeros_like(theta, dtype=np.complex64)
        for p in range(coeffs.shape[0]):
            for l in range(coeffs.shape[1]):
                result += coeffs[p, l] * modGLM_ff(theta, phi, z, self.k,
                                                   self.w0, p=p, l=l)
        return result
