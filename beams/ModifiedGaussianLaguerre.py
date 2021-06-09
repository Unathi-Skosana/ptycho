"""
File: ModifiedGaussianLaguerre.py
Author: PaulKGrimes
Github: https://github.com/PaulKGrimes/GaussianBeams
Description: 
"""


from math import factorial
from numpy import pi, sqrt, arctan, exp, sin, cos
from scipy.special import eval_genlaguerre
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


def alpha_nf(r, w, F):
    """
    Computes dimensionless parameter involving the radial distance and the beam
    radius for calculation convenience in the near field.

    Args:
        r: radial distance from the center axis of the beam
        w: radius at which the field amplitude fall to 1/e
        F: dimensionless parameter involving radial distance and beam curvature
    Returns:
        dimensionless parameter
    """

    return sqrt(2) * r / w / F


def alpha_ff(theta, w0, k):
    """
    Computes dimensionless parameter involving the radial distance and the beam
    radius for calculation convenience in the far field.

    Args:
        theta: radial distance from the center axis of the beam
        w0: waist radius
        k: wave number in the direction of propagation 
    Returns:
        dimensionless parameter
    """

    return sqrt(2) * k * w0 * sin(theta)

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


def amplitude_nf(r, z, k, w0, p=0, l=0):
    """
    Computes the near amplitude of a Gaussian-Laguerre beam

    Args:
        r: radial distance from the center axis of the beam
        z: axial distance from the beam waist
        k: wave number in the direction of propagation
        w0: beam waist radius
        l: degree of mode in the x direction
        m: degree of mode in the y direction
    Returns:
        near field amplitude of Gaussian-Hermite beam
    """

    z0 = rayleigh_range(w0, k)
    F = scale(r, z, z0)
    w = beam_rad(z, w0, z0)
    a = alpha_nf(r, w, F)

    return w0 / w * 1 / F**2 * a**abs(l) * lag_pl(a**2, p, l)


def amplitude_ff(theta, z, k, w0, p=0, l=0):
    """
    Computes the far amplitude of a Gaussian-Laguerre beam

    Args:
        r: radial distance from the center axis of the beam
        z: axial distance from the beam waist
        k: wave number in the direction of propagation
        w0: beam waist radius
        l: degree of mode in the x direction
        m: degree of mode in the y direction
    Returns:
        near field amplitude of Gaussian-Hermite beam
    """

    a = alpha_ff(theta, w0, k)
    z0 = rayleigh_range(w0, k)

    return z / z0 * cos(theta) * a**abs(l) * lag_pl(a**2, p, l)


def phase_nf(r, phi, z, k, w0, p=0, l=0):
    """
    Computes the near field phase of a Gaussian-Laguerre beam

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
        near field phase of Gaussian-Laguerre beam
    """

    z0 = rayleigh_range(w0, k)
    F = scale(r, z, z0)
    w = beam_rad(z, w0, z0)
    R = beam_curv(z, z0)
    a = alpha_nf(r, w, F)

    return exp(-a**2 / 2
               - j*k*R*(F - 1)
               - j * k * z
               + j * (2 * p + abs(l) +  1)
               * phi0(z, z0)
               - j * l * phi)


def phase_ff(theta, phi, z, k, w0, p=0, l=0):
    """
    Computes the far field phase of a Gaussian-Laguerre beam

    Args:
        theta: angle that parameterizes the far field i.e angle of boresight
        phi: azimuthal angle between the reference direction and projection
            of r onto the plane
        z: axial distance from the beam waist
        k: wave number in the direction of propagation
        w0: beam waist radius
        p: degree of generalized Laguerre polynomial
        l: integer in the Laguerre diff eq.
    Returns:
        far field phase of Gaussian-Laguerre beam
    """

    a = alpha_ff(theta, w0, k)

    return exp(- a**2 / 2
               + j * (2*p + abs(l) + 1) * pi / 2
               - j * l * z
               - j * k * phi)


def mod_gauss_lag_modes_nf(r, phi, z, k, w0, p=0, l=0):
    """
    Computes near field Gaussian-Laguerre mode with amplitude and phase 

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
        near field complex Gaussian-Laguerre mode
    """

    return amplitude_nf(r, z, k, w0, p, l) \
        * phase_nf(r, phi, z, k, w0, p, l)


def mod_gauss_lag_modes_ff(theta, phi, z, k, w0, p=0, l=0):
    """
    Computes far field Gaussian-Laguerre mode with amplitude and phase

    Args:
        theta: angle that parameterizes the far field i.e angle of boresight
        phi: azimuthal angle between the reference direction and projection
            of r onto the plane
        z: axial distance from the beam waist
        k: wave number in the direction of propagation
        w0: beam waist radius
        p: degree of generalized Laguerre polynomial
        l: integer in the Laguerre diff eq.
    Returns:
        far field complex Gaussian-Laguerre mode
    """

    return amplitude_ff(theta, z, k, w0, p, l) \
        * phase_ff(theta, phi, z, k, w0, p, l)
