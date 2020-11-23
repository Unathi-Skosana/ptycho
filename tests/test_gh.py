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

    beam = GHM(w0=80, k=5.0, maxL=6, maxM=6)
    zz = 5.0
    width = 80
    xx, yy = np.meshgrid(np.mgrid[-20:20:width*j], np.mgrid[-20:20:width*j])

    c = np.zeros(beam.shape)
    c[1, 2] = 1.0
    u = beam.field(xx, yy, zz, c)

    fig1, axes1 = plt.subplots(1, 2)
    ax1, ax2 = axes1.ravel()
    ax1.set_title(r'Amplitude')
    ax1.set_axis_off()
    ax1.imshow(np.abs(u), cmap='RdBu')
    ax2.set_title('Phase')
    ax2.set_axis_off()
    ax2.imshow(np.angle(u), cmap='RdBu')

    c = np.zeros(beam.shape)
    c[2, 3] = 1.0
    u = beam.field(xx, yy, zz, c)

    fig2, axes2 = plt.subplots(1, 2)
    ax3, ax4 = axes2.ravel()
    ax3.set_title(r'Amplitude')
    ax3.set_axis_off()
    ax3.imshow(np.abs(u), cmap='RdBu')
    ax4.set_title('Phase')
    ax4.set_axis_off()
    ax4.imshow(np.angle(u), cmap='RdBu')

    c = np.zeros(beam.shape)
    c[3, 4] = 1.0
    u = beam.field(xx, yy, zz, c)

    fig3, axes3 = plt.subplots(1, 2)
    ax5, ax6 = axes3.ravel()
    ax5.set_title(r'Amplitude')
    ax5.set_axis_off()
    ax5.imshow(np.abs(u), cmap='RdBu')
    ax6.set_title('Phase')
    ax6.set_axis_off()
    ax6.imshow(np.angle(u), cmap='RdBu')

    c = np.zeros(beam.shape)
    c[4, 5] = 1.0
    u = beam.field(xx, yy, zz, c)

    fig4, axes4 = plt.subplots(1, 2)
    ax7, ax8 = axes4.ravel()
    ax7.set_title(r'Amplitude')
    ax7.set_axis_off()
    ax7.imshow(np.abs(u), cmap='RdBu')
    ax8.set_title('Phase')
    ax8.set_axis_off()
    ax8.imshow(np.angle(u), cmap='RdBu')

    fig1.savefig('gh12.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig2.savefig('gh23.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig3.savefig('gh34.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig4.savefig('gh45.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)

    plt.show()
