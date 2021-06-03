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

    mod_beam = modGLM(w0=10e-3, k=10e9, maxP=6, maxL=6)
    width = 70
    zz_nf = 0.1
    zz_ff = 50
    xx, yy = np.meshgrid(np.mgrid[-10:10:width*j], np.mgrid[-10:10:width*j]);
    rr = np.sqrt(xx**2 + yy**2);
    theta = np.arcsin(rr / zz_ff)
    phi = np.arctan2(yy, xx);

    c = np.zeros(mod_beam.shape)
    c[1, 2] = 0.5
    u_nf = mod_beam.near_field(rr, phi, zz_nf, c)
    u_ff = mod_beam.far_field(theta, phi, zz_ff, c)

    fig1, axes1 = plt.subplots(1, 2)
    ax1, ax2 = axes1.ravel()
    ax1.set_title(r'Amplitude')
    ax1.set_axis_off()
    ax1.imshow(np.abs(u_ff), cmap='RdBu')
    ax2.set_title('Phase')
    ax2.set_axis_off()
    ax2.imshow(np.angle(u_ff), cmap='RdBu')

    c = np.zeros(mod_beam.shape)
    c[2, 3] = 1.0
    u_ff = mod_beam.near_field(rr, phi, zz_ff, c)
    u_ff = mod_beam.far_field(theta, phi, zz_ff, c)

    fig2, axes2 = plt.subplots(1, 2)
    ax3, ax4 = axes2.ravel()
    ax3.set_title(r'Amplitude')
    ax3.set_axis_off()
    ax3.imshow(np.abs(u_ff), cmap='RdBu')
    ax4.set_title('Phase')
    ax4.set_axis_off()
    ax4.imshow(np.angle(u_ff), cmap='RdBu')

    c = np.zeros(mod_beam.shape)
    c[3, 4] = 1.0
    u_ff = mod_beam.near_field(rr, phi, zz_ff, c)
    u_ff = mod_beam.far_field(theta, phi, zz_ff, c)

    fig3, axes3 = plt.subplots(1, 2)
    ax5, ax6 = axes3.ravel()
    ax5.set_title(r'Amplitude')
    ax5.set_axis_off()
    ax5.imshow(np.abs(u_ff), cmap='RdBu')
    ax6.set_title('Phase')
    ax6.set_axis_off()
    ax6.imshow(np.angle(u_ff), cmap='RdBu')

    c = np.zeros(mod_beam.shape)
    c[4, 5] = 1.0
    u_ff = mod_beam.near_field(rr, phi, zz_ff, c)
    u_ff = mod_beam.far_field(theta, phi, zz_ff, c)

    fig4, axes4 = plt.subplots(1, 2)
    ax7, ax8 = axes4.ravel()
    ax7.set_title(r'Amplitude')
    ax7.set_axis_off()
    ax7.imshow(np.abs(u_ff), cmap='RdBu')
    ax8.set_title('Phase')
    ax8.set_axis_off()
    ax8.imshow(np.angle(u_ff), cmap='RdBu')

    fig1.savefig('ff_gl12.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig2.savefig('ff_gl23.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig3.savefig('ff_gl34.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig4.savefig('ff_gl45.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)

    plt.show()
