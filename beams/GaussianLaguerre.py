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
    Computes the radius of the curvature of the beam wavefront at an axial distance of
    z from the beam's waist.

    Args:
        z: axial distance from beam's waist
        z0: rayleigh range
    Returns:
        radius of curvature
    """
    if z == 0:
        return np.Inf

    return z *(1 + (z0 / z)**2)


def beam_rad(z, w0, z0):
    """
    Computes the radius at which the field amplitude fall to 1\e of their
    axial values' at the plane z along the beam.

    Args:
        z: axial distance from beam's waist
        w0: waist radius
        z0: rayleigh range
    Returns:
        radius at which the amplitude fall to 1/e
    """
 
    return w0 * sqrt(1 + (z / z0)**2)


def rayleigh_range(w0, k):
    """
    Computes the rayleigh range, which is the distance along the propagation
    direction of a beam from the waist to the place where the area of cross
    section is doubled.
    
    Args:
        w0: waist radius of beam
        k: wave number in the direction of propagation of beam
    Returns:
        rayleigh range
    """

    return 0.5 * k * w0**2


def phi0(z, z0):
    """
    Computes the Gouy phase acquired by a beam at an axial distance of z

    Args:
        z: axial distance from beam's waist
        z0: rayleigh range
    Returns:
        gouy phase
    """

    return arctan(z / z0)


def alpha(r, w):
    """
    Computes dimensionless parameter involving the radial distance and the beam
    radius for calculation convenience.

    Args:
        r: radial distance from the center axis of the beam
        w: radius at which the field amplitude fall to 1/e
    Returns:
        dimensionless parameter
    """

    return sqrt(2) * r / w

def scale(r, z, z0):
    """
    Computes dimensionless parameter involving the radial distance and the beam
    curvature for calculation convenience in the near field. (F'' in  Tuovinen)

    Args:
        r: radial distance from the center axis of the beam
        z: axial distance from beam's waist
        z0: rayleigh range
    """

    R = beam_curv(z, z0)

    return sqrt(1 + (r / R)**2)

def lag_pl(x, p=0, l=0, normalize=True):
    """
    Evaluates the generalized Laguerre polynomial at x

    Args:
        x: value of evaluation
        p: degree of polynomial
        l: integer parameter in the Laguerre diff eq.
        normalize: normalize polynomial or not
    Returns:
        pth order generalized Laguerre polynomial at x
    """

    C_pl = sqrt(2 * factorial(p) / pi / factorial(p + abs(l)))

    if normalize:
        return C_pl * eval_genlaguerre(p, abs(l), x)
    return eval_genlaguerre(p, abs(l), x)


def amplitude(r, z, k, w0, p=0, l=0):
    """
    Computes the amplitude of a Gaussian-Laguerre beam

    Args:
        r: radial distance from the center axis of the beam
        z: axial distance from the beam waist
        k: wave number in the direction of propagation
        w0: beam waist radius
        l: degree of mode in the x direction
        m: degree of mode in the y direction
    Returns:
        amplitude of Gaussian-Hermite beam
    """
    z0 = rayleigh_range(w0, k)
    w = beam_rad(z, w0, z0)
    a = alpha(r, w)
    FF = scale(r, z, z0)
    a /= FF

    return np.array((1.0 + 1.0/FF) / 2.0 * 1/(k * w *FF) * a**abs(l) *
            lag_pl(a**2, p, abs(l)), dtype=np.float128)


def phase(r, phi, z, k, w0, p=0, l=0):
    """
    Computes the amplitude of a Gaussian-Laguerre beam

    Args:
        r: radial distance from the center axis of the beam
        phi: azimuthal angle between the reference direction and projection
            of r onto the plane
        z: axial distance from the beam waist
        k: wave number in the direction of propagation
        w0: beam waist radius
        p: degree of generalized Laguerre polynomial
        l: integer in the Laguerre diff eq.
    Returns:
        phase of Gaussian-Laguerre beam
    """

    z0 = rayleigh_range(w0, k)
    w = beam_rad(z, w0, z0)
    R = beam_curv(z, z0)
    a = alpha(r, w)
    FF = scale(r, z, z0)
    a /= FF

    return np.array(exp(-a**2 / 2.0
               - 1j * k * R * (FF - 1)
               - 1j * k * z
               + 1j * (2 * p + abs(l) + 1) * phi0(z, z0)
               - 1j * l * phi), dtype=np.complex128)


def gauss_lag_modes(r, phi, z, k, w0, p=0, l=0):
    """
    Computes Gaussian-Laguerre mode with amplitude and phase

    Args:
        r: radial distance from the center axis of the beam
        phi: azimuthal angle between the reference direction and projection
            of r onto the plane
        z: axial distance from the beam waist
        k: wave number in the direction of propagation
        w0: beam waist radius
        p: degree of generalized Laguerre polynomial
        l: integer in the Laguerre diff eq.
    Returns:
        complex Gaussian-Laguerre mode
    """

    return amplitude(r, z, k, w0, p, l) * \
        phase(r, phi, z, k, w0, p, l)
