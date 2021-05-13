"""
File: filters.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
"""

import numpy as np
from scipy import signal
from numpy import pi


def gau_kern(dim, std, normalize=False):
    '''
    Generates a dim x dim matrix with a centered Gaussian
    of standard deviation std centered on it. If normalized,
    its volume equals 1.

    Source:
        https://gist.github.com/thomasaarholt/267ec4fff40ca9dff1106490ea3b75670

    Args:
        dim: dimension of square matrix
        std: standard deviation of Gaussian
        normalize: Normalize Gaussian

    Returns:
        Gaussian matrix kernel
    '''

    gauss_1d = signal.gaussian(dim, std)
    gauss_2d = np.outer(gauss_1d, gauss_1d)

    if normalize:
        gauss_2d /= (2 * pi * (std**2))
    return gauss_2d


def circ_mask(dim, center, radius, val=1, inverse=True):
    '''
    Generates an dim x dim circular mask centered at c with radius r

    Args:
        dim: dimension of square matrix mask
        center: center of circle
        radius: radius of circle
        val: value of mask
    Returns:
        Circular mask
    '''

    c_x, c_y = center
    g_y, g_x = np.ogrid[-c_x:dim-c_x, -c_y:dim-c_y]
    mask = g_x*g_x + g_y*g_y <= radius*radius
    arr = None
    if inverse:
        arr = np.full((dim, dim), val)
        arr[mask] = 0
    else:
        arr = np.zeros((dim, dim))
        arr[mask] = val
    return arr
