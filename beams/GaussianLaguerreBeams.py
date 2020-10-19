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
        # Create a complex array holding the mode coefficients of the G-L modes
        # Indexing runs from p=0 to p=maxP in the first dimension and
        # l=0 to maxL then -maxL to l=-1 in the second.

        self._coeffs = np.full((maxP+1, 2*maxL+1), complex(1, 0),dtype=complex)

        self._k = k    # The wavenumber of the G-L modes
        self._w0 = w0  # The beam waist radius of the G-L modes
        self._maxP = maxP  # The highest index of the axial modes included in the modeset
                           # axial mode index p is in the range 0 < p < maxP
        self._maxL = maxL  # The highest absolute index of the azimuthal modes included in the  modeset
                           # azimuthal mode index l is in the range -maxL < l < maxL
        self.resizeModeSet(maxP, maxL)

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

    def resizeModeSet(self, p, l):
        """
        Resize the array of mode coefficients
        """

        # Don't have to do anything clever with p indices

        self._coeffs.resize(p+1, self._maxL*2+1)
        self._maxP = p

        # Have to be clever with l indices to get correct array shape
        if l > self._maxL:
            # adding 2*(l-maxL) columns in middle of array
            # first column to add becomes maxL+1
            # last column added becomes l*2+1 - (maxL+1) = l*2-maxL
            # first column to move to end is maxL+1
            # last column to move to end is 2*maxL+1
            fstColSrc = self._maxL+1
            lstColSrc = 2*self._maxL+1
            fstColDest = 2*l - self._maxL + 1
            newCoeffs = np.full((self._maxP+1, l*2+1), complex(1, 0), dtype=complex)
            newCoeffs[:, :fstColSrc] = self._coeffs[:, :fstColSrc]
            newCoeffs[:, fstColDest:] = self._coeffs[:, fstColSrc:lstColSrc]
            self._coeffs = newCoeffs
        if l < self._maxL:
            # adding 2*(l-maxL) columns in middle of array
            # first column to move is 2*maxL+1-l
            # last column to move is  2*maxL+1
            # first column to move moves to l+1
            # last column to move moves to 2*l+1

            fstColSrc = 2*self._maxL+1-l
            lstColSrc = 2*self._maxL+1
            fstColDest = l+1
            newCoeffs = np.full((self._maxP+1, l*2+1), complex(1, 0), dtype=complex)
            if l > 0:
                newCoeffs[:, :l+2] = self._coeffs[:, :l+2]
                newCoeffs[:, fstColDest:] = \
                    self._coeffs[:, fstColSrc:lstColSrc]
            else:
                # special case if we are dropping down to
                # only the fundamental azimuthal mode
                newCoeffs[:, 0] = self._coeffs[:, 0]
            self._coeffs = newCoeffs
        self._maxL = l

    @property
    def maxP(self):
        """
        return the maximum absolute index for the axial mode index
        """

        return self._maxP

    @maxP.setter
    def maxP(self, p):
        """
        Set a new value for maxP
        """

        # resize the self._coeffs array

        self.resizeModeSet(p, self._maxL)

    @property
    def maxL(self):
        """
        return the maximum absolute index for the azimuthal mode index
        """

        return self._maxL

    @maxL.setter
    def maxL(self, l):
        """
        Set a new value for maxL
        """
        # resize the self._coeffs array

        self.resizeModeSet(self._maxP, l)


class GaussLaguerreModeSet(GaussLaguerreModeBase):
    """
    A class holding a set of Gauss-Laguerre modes, defined in
    the paraxial limit.
    """

    def __init__(self, w0=1., k=1., maxP=0, maxL=0):
        super(GaussLaguerreModeSet, self).__init__(w0, k, maxP, maxL)

    def field(self, r, phi, z, p=None, l=None):
        """
        Return the value of the field at r, phi, z; either for the sum of
        all modes (p, l) = None, for a specified axial mode
        p (sum over azimuthal modes), or for a specific (p, l) mode.
        """

        if p is not None and l is not None:
            # We are after a specific mode
            return self._coeffs[p, l] * GLM(r, phi, z,
                                            self.w0, self.k, p=p, l=l)
        elif p is not None and l is None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(r, dtype=np.complex)
            for ll in range(-self._maxL, self._maxL+1):
                result += self.field(r, phi, z, p=p, l=ll)
            return result
        elif p is None and l is None:
            # We are after the sum of all modes.
            result = np.zeros_like(r, dtype=np.complex)
            for pp in range(0, self._maxP+1):
                result += self.field(r, phi, z, p=pp, l=None)
            return result

        raise ValueError("GaussLaguerreModeSet.field: must set mode index p \
                          if mode index l is set")


class ModifiedGaussLaguerreModeSet(GaussLaguerreModeBase):
    """
    A class holding a set of modified Gauss-Laguerre modes,
    using the definition Tuovinen (1992)
    """

    def __init__(self, w0=1., k=1., maxP=0, maxL=0):
        super(ModifiedGaussLaguerreModeSet, self).__init__(w0, k, maxP, maxL)

    def near_field(self, r, phi, z, p=None, l=None):
        """
        Return the value of the field at r, phi, z; either for the sum of all
        modes (p, l) = None, for a specified axial mode 
        p (sum over azimuthal modes), or for a specific (p, l) mode.
        """

        if p is not None and l is not None:
            # We are after a specific mode
            return self._coeffs[p, l] * modGLM_nf(r, phi, z,
                                                  self.w0, self.k,
                                                  p=p, l=l)
        elif p is not None and l is None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(r, dtype=np.complex)
            for ll in range(-self.maxL, self.maxL+1):
                result += self.near_field(r, phi, z, p=p, l=ll)
            return result
        elif p is None and l is None:
            # We are after the sum of all modes.
            result = np.zeros_like(r, dtype=np.complex)
            for pp in range(0, self.maxP+1):
                result += self.near_field(r, phi, z, p=pp, l=None)
            return result

        raise ValueError("ModifiedGaussLaguerreModeSet.near_field: must set mode \
                index p if mode index l is set")

    def far_field(self, theta, phi, z, p=None, l=None):
        """
        Return the value of the far field at theta, phi; either for the sum of all
        modes (p, l) = None, for a specified axial mode
        p (sum over azimuthal modes), or for a specific (p, l) mode.
        """

        if p is not None and l is not None:
            # We are after a specific mode
            return self._coeffs[p, l] * modGLM_ff(theta, phi, z,
                                                self.w0, self.k,
                                                p=p, l=l)
        elif p is not None and l is None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(theta, dtype=np.complex)
            for ll in range(-self.maxL, self.maxL+1):
                result += self.far_field(theta, phi, z, p=p, l=ll)
            return result
        elif p is None and l is None:
            # We are after the sum of all modes.
            result = np.zeros_like(theta, dtype=np.complex)
            for pp in range(0, self.maxP+1):
                result += self.far_field(theta, phi, z, p=pp, l=None)
            return result

        raise ValueError("ModifiedGaussLaguerreModeSet.far_field: must set \
                          mode index p if mode index l is set")
