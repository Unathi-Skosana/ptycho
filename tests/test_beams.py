"""
File: test_beams.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
"""

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    from beams.GaussianLaguerreBeams import GaussLaguerreModeSet as GLM
    from common.constants import c, pi, j

    plt.style.use('mint')

    beam = modGLM(w0=.1, k=1.0, maxP=6, maxL=6)

    width = 80
    z = 1.0

    xx, yy = np.meshgrid(np.mgrid[-10:10:width*j], np.mgrid[-10:10:width*j]);

    # Calculate the cylindrical coordinates
    r = np.sqrt(xx**2 + yy**2);
    phi = np.arctan2(yy, xx);

    u = beam.near_field(r, phi, z, p=0, l=2)

    width, height = plt.figaspect(1.25)

    fig1, ax1 = plt.subplots(figsize=(width, height), dpi=96)
    fig2, ax2 = plt.subplots(figsize=(width, height), dpi=96)

    ax1.set_title('Field amplitude')
    ax1.imshow(np.abs(u), cmap='gray')

    ax2.set_title('Field phase')
    ax2.imshow(np.angle(u), cmap='gray')

    plt.show()
