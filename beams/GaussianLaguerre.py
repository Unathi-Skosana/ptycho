"""
File: GaussianLaguerre.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
    Adapted from https://github.com/PaulKGrimes/GaussianBeams/blob/master/GaussianLaguerreModes.py
"""


from math import factorial
from numpy import pi,  sqrt, arctan, exp
from scipy.special import eval_genlaguerre
from common.constants import j, abs_A

import numpy as np


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


def lag_pl(x, p=0, l=0, normalize=True):
    """
    return the value of the Gaussian-Laguerre polynomial at x
    """

    C_pl = sqrt(2 * factorial(p) / pi / factorial(p + abs(l)))

    if normalize:
        return C_pl * eval_genlaguerre(p, abs(l), x)
    return eval_genlaguerre(p, abs(l), x)


def amplitude(r, z, k, w0, p=0, l=0):
    """
    docs
    """
    z0 = rayleight_range(w0, k)
    w = beam_rad(z, w0, z0)
    a = alpha(r, w)

    return w0 / w * a**abs(l) * lag_pl(a**2, p, l)


def longitude(r, phi, z, k, w0, p, l):
    """
    docs
    """

    z0 = rayleight_range(w0, k)
    w = beam_rad(z, w0, z0)
    R = beam_curv(z, z0)
    a = alpha(r, w)

    return exp(- a**2 / 2 - j * k * r**2 / 2 / R \
            - j * k * z \
            + (2 * p + abs(l) + 1) * phi0(z, z0) \
            - j * l * phi)


def gauss_lag_modes(r, phi, z, k, w0, p=0, l=0):
    """
    docs
    """

    return amplitude(r, z, k, w0, p, l) * longitude(r, phi, z, k, w0, p, l)
