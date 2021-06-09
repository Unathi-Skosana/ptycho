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

    from beams.GaussianLaguerreBeams import ModifiedGaussLaguerreModeSet as modGLM
    from common.constants import c, pi, j
    from utils.filters import gau_kern

    plt.style.use('mint')

    wavelength = 624*1e-9
    k = 2 * np.pi / wavelength
    grid_size = 30/ wavelength
    w0 = 100 * wavelength
    N = 70
    dx = grid_size / N

    w = N
    cx = int(w/2)

    h = N
    cy = int(h/2)

    Y, X = np.mgrid[:h, :w]
    Y = (Y-cy) * dx
    X = (X-cx) * dx

    z_nf = 1*1e-9
    z_ff = 5*1e9

    mod_beam = modGLM(w0=w0, k=k, maxP=6, maxL=6)

    r = np.sqrt(X**2 + Y**2);
    theta = np.arcsin(r / z_ff)
    phi = np.arctan2(Y, X);

    c = np.zeros(mod_beam.shape)
    c[2, 2] = 1.0

    u_nf = mod_beam.near_field(r, phi, z_nf, c)
    u_ff = mod_beam.far_field(theta, phi, z_ff, c)

    fig1, axes1 = plt.subplots(1, 2)
    ax1, ax2 = axes1.ravel()

    ax1.set_title(r'Amplitude')
    ax1.set_axis_off()
    ax1.imshow(np.abs(u_ff), cmap='RdBu')

    ax2.set_title('Phase')
    ax2.set_axis_off()
    ax2.imshow(np.angle(u_ff), cmap='RdBu')

    plt.show()
