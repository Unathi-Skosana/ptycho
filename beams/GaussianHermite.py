"""
File: GaussianHermite.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
    Adapted from https://github.com/PaulKGrimes/GaussianBeams/blob/master/GaussianLaguerreModes.py
"""

from math import factorial
from numpy import pi, sqrt, arctan, exp
from scipy.special import eval_hermite

import numpy as np

from common.constants import j, abs_A

def beam_curv(z, z0):
    """
    doc
    """
    return z + z0**2 / z


def beam_rad(z, w0, z0):
    """
    doc
    """
    return w0 * sqrt(1 + (z/z0)**2)


def rayleight_range(w0, k):
    """
    doc
    """
    return k * w0**2


def phi0(z, z0):
    """
    doc
    """
    return arctan(z / z0)


def alpha(r, w):
    """
    doc
    """

    return np.sqrt(2) * r / w


def herm_n(x, n=0):
    """
    return the value of the Gaussian-Laguerre polynomial at x
    """

    return eval_hermite(n, x)


def amplitude(x, y, z, k, w0, l=0, m=0):
    """
    docs
    """
    z0 = rayleight_range(w0, k)
    w = beam_rad(z, w0, z0)
    a_x = alpha(x, w)
    a_y = alpha(y, w)

    return w0 / w * herm_n(a_x, n=l) * herm_n(a_y, n=m)


def longitude(x, y, z, k, w0, l=0, m=0):
    """
    docs
    """

    r = sqrt(x**2 + y**2)
    z0 = rayleight_range(w0, k)
    w = beam_rad(z, w0, z0)
    R = beam_curv(z, z0)
    a = alpha(r, w)

    return exp(- a**2 / 2 - j * k * r**2 / 2 / R \
            - j * k * z \
            + (l + m + 1) * phi0(z, z0))



def gauss_herm_modes(x, y, z, k, w0, l=0, m=0):
    """
    docs
    """

    return amplitude(x, y, z, k, w0, l, m) * longitude(x, y, z, k, w0, l, m)
