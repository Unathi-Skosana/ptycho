"""
File: test_beams.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
"""


import numpy as np
import matplotlib.pyplot as plt

from beams.GaussianLaguerreBeams import GaussLaguerreModeSet as GLM
from common.constants import c, pi, j


if __name__ == '__main__':

    plt.style.use('mint')

    beam = GLM(w0=1.0, k=1.0, maxP=6, maxL=6)

    width = 256
    z = 2.0

    xx, yy = np.meshgrid(np.mgrid[-20:20:width*j], np.mgrid[-20:20:width*j]);

    # Calculate the cylindrical coordinates
    r = np.sqrt(xx**2 + yy**2);
    phi = np.arctan2(yy, xx);

    u = beam.field(r, phi, z, p=3, l=0)

    width, height = plt.figaspect(1.25)

    fig1, ax1 = plt.subplots(figsize=(width, height), dpi=96)
    fig2, ax2 = plt.subplots(figsize=(width, height), dpi=96)

    ax1.set_title('Field amplitude')
    ax1.imshow(np.abs(u)**2, cmap='gray')

    ax2.set_title('Field phase')
    ax2.imshow(np.angle(u)**2, cmap='gray')

    plt.show()
