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

    from common.constants import j
    from beams.GaussianHermiteBeams import GaussHermiteModeSet as GHM

    plt.style.use('mint')

    grid_size = 1*10e6
    wavelength = 624*10e-9
    w0 = 3*10e-6
    N = 256
    dx = grid_size / N

    w = N
    cx = int(w/2)
    xvals = dx * np.arange(-cx, (w-cx))

    h = N
    cy = int(h/2)
    yvals = dx * np.arange(-cy, (h-cy))

    Y, X = np.mgrid[:h, :w]
    Y = (Y-cy) * dx
    X = (X-cx) * dx

    z = 1*10e-6

    beam = GHM(w0=w0, k=2 * np.pi / wavelength, maxL=6, maxM=6)

    c = np.zeros(beam.shape)
    c[2, 1] = 1.0

    u = beam.field(X, Y, z, c)

    fig1, axes1 = plt.subplots(1, 2)
    ax1, ax2 = axes1.ravel()

    ax1.set_title(r'Amplitude')
    ax1.set_axis_off()
    ax1.imshow(np.abs(u), cmap='RdBu')

    ax2.set_title('Phase')
    ax2.set_axis_off()
    ax2.imshow(np.angle(u), cmap='RdBu')

    plt.show()
