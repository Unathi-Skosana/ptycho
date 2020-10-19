"""
File: ModifiedGaussianLaguerre.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
    Adapted from https://github.com/PaulKGrimes/GaussianBeams/blob/master/GaussianLaguerreModes.py
"""


from math import factorial
from numpy import pi, sqrt, arctan, exp, sin, cos
from scipy.special import eval_genlaguerre
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

    return w0 * sqrt(1 + (z / z0)**2)


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


def alpha_nf(r, w, F):
    """
    doc
    """

    return sqrt(2) * r / w / F


def alpha_ff(theta, w0, k):
    """
    doc
    """

    return sqrt(2) * k * w0 * sin(theta)


def scale(r, z, z0):
    """
    doc
    """

    R = beam_curv(z, z0)

    return sqrt(1 + (r / R)**2)


def lag_pl(x, p=0, l=0, normalize=True):
    """
    return the value of the Gaussian-Laguerre polynomial at x
    """

    C_pl = sqrt(2 * factorial(p) / pi / factorial(p + abs(l)))

    if normalize:
        return C_pl * eval_genlaguerre(p, abs(l), x)

    return eval_genlaguerre(p, abs(l), x)


def amplitude_nf(r, z, k, w0, p=0, l=0):
    """
    docs
    """
    z0 = rayleight_range(w0, k)
    F = scale(r, z, z0)
    w = beam_rad(z, w0, z0)
    a = alpha_nf(r, w, F)

    return w0 / w * 1 / F**2 * a**abs(l) * lag_pl(a**2, p, l)


def amplitude_ff(theta, z, k, w0, p=0, l=0):
    """
    docs
    """
    a = alpha_ff(theta, w0, k)
    z0 = rayleight_range(w0, k)

    return z / z0 * cos(theta) * a**abs(l) * lag_pl(a**2, p, l)


def longitude_nf(r, phi, z, k, w0, p, l):
    """
    docs
    """

    z0 = rayleight_range(w0, k)
    F = scale(r, z, z0)
    w = beam_rad(z, w0, z0)
    R = beam_curv(z, z0)
    a = alpha_nf(r, w, F)

    return exp(-a**2 / 2  \
            - j*k*R*(F - 1) \
            - j * k * z \
            + j * (2 * p + abs(l) +  1) * phi0(z, z0) \
            - j * l * phi)


def longitude_ff(theta, phi, z, k, w0, p=0, l=0):
    """
    docs
    """

    a = alpha_ff(theta, w0, k)

    return exp(- a**2 / 2 \
            + j * (2*p + abs(l) + 1) * pi / 2  \
            - j * l * z  \
            - j * k * phi)


def mod_gauss_lag_modes_nf(r, phi, z, k, w0, p=0, l=0):
    """
    docs
    """

    return amplitude_nf(r, z, k, w0, p, l) \
        * longitude_nf(r, phi, z, k, w0, p, l)


def mod_gauss_lag_modes_ff(theta, phi, z, k, w0, p=0, l=0):
    """
    docs
    """

    return amplitude_ff(theta, z, k, w0, p, l) \
        * longitude_ff(theta, phi, z, k, w0, p, l)