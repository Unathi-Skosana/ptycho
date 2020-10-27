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
    zz_nf = 1.0
    zz_ff = 1000
    xx, yy = np.meshgrid(np.mgrid[-5:5:width*j], np.mgrid[-5:5:width*j]);
    rr = np.sqrt(xx**2 + yy**2);
    theta = np.arcsin(rr / zz_ff)
    phi = np.arctan2(yy, xx);

    c = np.zeros(mod_beam.shape)
    c[0][0] = 1.0

    u_nf = mod_beam.near_field(rr, phi, zz_nf, c)
    u_ff = mod_beam.far_field(theta, phi, zz_ff, c)

    width, height = plt.figaspect(1.25)

    fig1, ax1 = plt.subplots(figsize=(width, height), dpi=96)
    fig2, ax2 = plt.subplots(figsize=(width, height), dpi=96)

    fig3, ax3 = plt.subplots(figsize=(width, height), dpi=96)
    fig4, ax4 = plt.subplots(figsize=(width, height), dpi=96)

    ax1.set_title('Near field amplitude')
    ax1.imshow(np.abs(u_nf)**2, cmap='RdBu')

    ax2.set_title('Near field phase')
    ax2.imshow(np.angle(u_nf)**2, cmap='RdBu')

    ax3.set_title('Far field amplitude')
    ax3.imshow(np.abs(u_ff)**2, cmap='RdBu')

    ax4.set_title('Far field phase')
    ax4.imshow(np.angle(u_ff)**2, cmap='RdBu')

    plt.show()
