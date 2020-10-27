"""
File: GaussianHermite.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/ptycho
Description:

Adapted from https://github.com/PaulKGrimes/GaussianBeams
"""

from math import factorial
from numpy import pi, sqrt, arctan, exp
from scipy.special import eval_hermite

import numpy as np

from common.constants import j, abs_A

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
    return z + z0**2 / z


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
    return w0 * sqrt(1 + (z/z0)**2)


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
    return k * w0**2


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

    return np.sqrt(2) * r / w


def herm_n(x, n=0):
    """
    Evaluates the nth Hermite polynomial at x

    Args:
        x: value of evaluation
        n: degree of polynomial
    Returns:
        nth Hermite polynomial at x
    """

    return eval_hermite(n, x)


def amplitude(x, y, z, k, w0, l=0, m=0):
    """
    Computes the amplitude of a Gaussian-Hermite beam

    Args:
        x: distance from the center axis of the beam in the x-direction
        y: distance from the center axis of the beam in the y-direction
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
    a_x = alpha(x, w)
    a_y = alpha(y, w)

    return w0 / w * herm_n(a_x, n=l) * herm_n(a_y, n=m)


def phase(x, y, z, k, w0, l=0, m=0):
    """
    Computes the phase of a Gaussian-Hermite beam

    Args:
        x: distance from the center axis of the beam in the x-direction
        y: distance from the center axis of the beam in the y-direction
        z: axial distance from the beam waist
        k: wave number in the direction of propagation
        w0: beam waist radius
        l: degree of mode in the x direction
        m: degree of mode in the y direction
    Returns:
        phase of Gaussian-Hermite beam
    """

    r = sqrt(x**2 + y**2)
    z0 = rayleigh_range(w0, k)
    w = beam_rad(z, w0, z0)
    R = beam_curv(z, z0)
    a = alpha(r, w)

    return exp(- a**2 / 2 - j * k * r**2 / 2 / R \
               - j * k * z \
               + (l + m + 1) * phi0(z, z0))


def gauss_herm_modes(x, y, z, k, w0, l=0, m=0):
    """
    Computes a Gaussian-Hermite modes

    Args:
        x: distance from the center axis of the beam in the x-direction
        y: distance from the center axis of the beam in the y-direction
        z: axial distance from the beam waist
        k: wave number in the direction of propagation
        w0: beam waist radius
        l: degree of mode in the x direction
        m: degree of mode in the y direction
    Returns:
        Gaussian-Hermite mode
    """

    return amplitude(x, y, z, k, w0, l, m) * phase(x, y, z, k, w0, l, m)
