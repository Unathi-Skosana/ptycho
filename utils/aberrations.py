"""
File: abberations.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
"""

from operator import itemgetter
from numpy import pi
from skimage.transform import resize
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from zernike import RZern

import numpy as np

from .formatting import pad_with


def zernike_poly(vals, radius, width):
    """
    Generates a w x w matrix of a linear combination of zernike
    polynomials from specified indices and coefficients

    Args:
        vals: list of tuples, with each tuple specifying the indices of the
            specific polynomial and it's coefficient
        r: radius of polynomial
        w: dimension of matrix
    Returns:
        Linear combination of zernike polynomials as matrix
    """

    # largest n index
    n_max = max(vals, key=itemgetter(1))[1]

    cart = RZern(n_max)
    xxv, yyv = np.meshgrid(np.mgrid[-1:1:width*1j], np.mgrid[-1:1:width*1j])
    cart.make_cart_grid(xxv, yyv)

    coeffs = np.zeros(cart.nk)

    for val in vals:
        noll_m, noll_n, coeff = val

        # uses Noll's sequential indices
        j = int(noll_n * (noll_n + 1) / 2 + np.abs(noll_m))

        if noll_m >= 0 and noll_n % 4 == 2 or noll_n % 4 == 3:
            j += 1
        if noll_m <= 0 and noll_n % 4 == 0 or noll_n % 4 == 1:
            j += 1
        coeffs[j-1] = coeff

    # convert nan to 0
    phi = np.nan_to_num(cart.eval_grid(coeffs, matrix=True), nan=0)

    # pad image with zeros outside the radius
    if width != radius:
        phi = np.pad(phi, width // 2 - radius // 2, pad_with, padder=0)

    # resize if necessary
    return resize(phi, (width, width), anti_aliasing=False)


def c_lpf(img, pixel_size, cutoff_freq):
    """
    Implements a coherent low pass filter in the spatial-frequency
    domain

    Args:
        img: input image
        pixel_size: pixel size of detector
        cutoff_freq: cut off frequency
    Returns:
        Coherently low pass filtered image
    """

    height, width = img.shape
    k_x = np.arange(-pi/pixel_size, pi/pixel_size, 2*pi / pixel_size / width)
    k_y = np.arange(-pi/pixel_size, pi/pixel_size, 2*pi / pixel_size / height)
    k_i, k_j = np.meshgrid(k_x, k_y)

    # cutoff frequency to implement low pass filter transfer function
    freq = k_i**2 + k_j**2
    ctf = freq <= cutoff_freq**2
    out_img = np.multiply(img, ctf)

    return out_img, ctf.astype(np.uint)


def inc_lpf(img, pixel_size, cutoff_freq):
    """
    Implements an incoherent low pass filter in the spatial-frequency domain

    Args:
        img: input image
        pixel_size: pixel size of detector
        cutoff_freq: cutoff frequency
    Returns:
        Incoherently low pass filter image
    """

    _, ctf = c_lpf(img, pixel_size, cutoff_freq)
    cpsf = fftshift(ifft2(ifftshift(ctf)))
    ipsf = np.abs(cpsf)**2
    otf = np.abs(fftshift(fft2(ifftshift(ipsf))))
    otf /= np.max(otf)
    out_img = np.multiply(img, otf)

    return out_img, otf


def c_abber(img, phi, pixel_size, cutoff_freq):
    """
    Introduces coherent aberrations to image via a low pass filter and
    some aberration wavefront phi

    Args:
        in_im: input image
        phi: aberration wavefront
        pixel_size: pixel size of detector
        cutoff_freq: cutoff frequency for low pass filter

    Returns:
        Coherently aberrated image
    """
    _, ctf = c_lpf(img, pixel_size, cutoff_freq)
    aber_ctf = np.multiply(np.exp(1j * phi), ctf)
    out_img = np.multiply(img, aber_ctf)

    return out_img, aber_ctf


def inc_abber(img, phi, pixel_size, cutoff_freq):
    """
    Introduces incoherent aberrations to image via a low pass filter
    and some aberration wavefront phi

    Args:
        img: input image
        phi: aberration wavefront
        pixel_size: pixel size of detector
        cutoff_freq: cutoff frequency for low pass filter

    Returns:
        Incoherently aberrated image
    """

    _, aber_ctf = c_abber(img, phi, pixel_size, cutoff_freq)
    cpsf = fftshift(ifft2(ifftshift(aber_ctf)))
    ipsf = np.abs(cpsf)**2
    otf = np.abs(fftshift(fft2(ifftshift(ipsf))))
    otf /= np.max(otf)
    out_img = np.multiply(img, otf)

    return out_img, otf
