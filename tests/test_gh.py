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

    from beams.GaussianHermiteBeams import GaussHermiteModeSet as GHM
    from common.constants import c, pi, j

    plt.style.use('mint')

    beam = GHM(w0=.1, k=1.0, maxL=6, maxM=6)
    width = 80

    zz = 1.0
    xx, yy = np.meshgrid(np.mgrid[-10:10:width*j], np.mgrid[-10:10:width*j]);

    # Calculate the cylindrical coordinates
    phi = np.arctan2(yy, xx);

    c = np.zeros(beam.shape)
    c[2][0] = 1.0

    u = beam.field(xx, yy, zz, c)

    width, height = plt.figaspect(1.25)

    fig1, ax1 = plt.subplots(figsize=(width, height), dpi=96)
    fig2, ax2 = plt.subplots(figsize=(width, height), dpi=96)

    ax1.set_title('Field amplitude')
    ax1.imshow(np.abs(u), cmap='RdBu')

    ax2.set_title('Field phase')
    ax2.imshow(np.angle(u), cmap='RdBu')

    plt.show()
