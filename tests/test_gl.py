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
    from skimage.restoration import unwrap_phase

    from common.constants import j
    from beams.GaussianLaguerreBeams import GaussLaguerreModeSet as GLM

    plt.style.use('mint')

    grid_size = 10e6
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

    zz = 1*10e-6

    beam = GLM(w0=w0, k = 2 * np.pi / wavelength, maxP=6, maxL=6)
    rr = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    c = np.zeros(beam.shape)
    c[2, 8]= 1
    u = beam.field(rr, phi, zz, c)

    fig1, axes1 = plt.subplots(1, 2)
    ax1, ax2 = axes1.ravel()

    ax1.set_title(r'Amplitude')
    ax1.set_axis_off()
    ax1.imshow(np.abs(u), cmap='RdBu')

    ax2.set_title('Phase')
    ax2.set_axis_off()
    ax2.imshow(unwrap_phase(np.angle(u)), cmap='RdBu')

    plt.show()
