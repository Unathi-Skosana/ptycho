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


    plt.style.use('mint')

    mod_beam = modGLM(w0=.5, k=.5, maxP=6, maxL=6)

    width = 80
    z_nf = 1.0
    z_ff = 1000

    xx, yy = np.meshgrid(np.mgrid[-2:2:width*j], np.mgrid[-2:2:width*j]);

    # Calculate the cylindrical coordinates
    r = np.sqrt(xx**2 + yy**2);
    theta = np.arcsin(r / z_ff)
    phi = np.arctan2(yy, xx);

    u_nf = mod_beam.near_field(r, phi, z_nf, p=3, l=0)
    u_ff = mod_beam.far_field(theta, phi, z_ff, p=3, l=0)

    width, height = plt.figaspect(1.25)

    fig1, ax1 = plt.subplots(figsize=(width, height), dpi=96)
    fig2, ax2 = plt.subplots(figsize=(width, height), dpi=96)

    fig3, ax3 = plt.subplots(figsize=(width, height), dpi=96)
    fig4, ax4 = plt.subplots(figsize=(width, height), dpi=96)

    ax1.set_title('Near field amplitude')
    ax1.imshow(np.abs(u_nf)**2, cmap='gray')

    ax2.set_title('Near field phase')
    ax2.imshow(np.angle(u_nf)**2, cmap='gray')

    ax3.set_title('Far field amplitude')
    ax3.imshow(np.abs(u_ff)**2, cmap='gray')

    ax4.set_title('Far field phase')
    ax4.imshow(np.angle(u_ff)**2, cmap='gray')

    plt.show()
