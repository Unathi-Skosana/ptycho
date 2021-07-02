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
    from skimage.exposure import rescale_intensity
    from mpl_toolkits.axes_grid1 import ImageGrid

    from common.constants import j
    from beams.GaussianLaguerreBeams import GaussLaguerreModeSet as GLM
    from utils.filters import circ_mask

    plt.style.use('mint')

    wavelength = 624 * 1e-9
    k = 2 * np.pi / wavelength
    w0 = 20 * 1e-6

    N = 70
    L = 100 * 1e-6
    x0 = np.linspace(-L/2, L/2, N)
    y0 = np.linspace(-L/2, L/2, N)
    xv, yv = np.meshgrid(x0, y0)
    zz = 1*1e-9

    print((x0[0] - x0[1]) * 1e6 * N)

    beam = GLM(w0=w0, k=k, maxP=8, maxL=8)
    rr = np.sqrt(xv**2 + yv**2)
    phi = np.arctan2(yv, xv)

    cmask = circ_mask(N, (N//2, N//2), N//2, 1.0, inverse=False)

    c = np.zeros(beam.shape)
    c[0, 0] = 1.0
    u1 = beam.field(rr, phi, zz, c)
    c[0, 0] = 0.0
    c[1, 0] = 1.0
    u2 = beam.field(rr, phi, zz, c)
    c[1, 0] = 0.0
    c[2, 0] = 1.0
    u3 = beam.field(rr, phi, zz, c)
    c[2, 0] = 0.0
    c[0, 1] = 1.0
    u4 = beam.field(rr, phi, zz, c)
    c[0, 1] = 0.0
    c[1, 1] = 1.0
    u5 = beam.field(rr, phi, zz, c)
    c[1, 1] = 0.0
    c[2, 1] = 1.0
    u6 = beam.field(rr, phi, zz, c)
    c[2, 1] = 0.0
    c[0, 2] = 1.0
    u7 = beam.field(rr, phi, zz, c)
    c[0, 2] = 0.0
    c[1, 2] = 1.0
    u8 = beam.field(rr, phi, zz, c)
    c[1, 2] = 0.0
    c[2, 2] = 1.0
    u9 = beam.field(rr, phi, zz, c)

    fig = plt.figure(figsize=(25, 25))
    grid = ImageGrid(fig,
            111,  # similar to subplot(111)
            nrows_ncols=(3, 6),  # creates 2x2 grid of axes
            axes_pad=0.05,  # pad between axes in inch.
    )

    i = 0
    for ax, im in zip(grid, [np.abs(u1), np.angle(u1), np.abs(u2), np.angle(u2),
        np.abs(u3), np.angle(u3), np.abs(u4), np.angle(u4), np.abs(u5),
        np.angle(u5), np.abs(u6), np.angle(u6), np.abs(u7), np.angle(u7),
        np.abs(u8), np.angle(u8), np.abs(u9), np.angle(u9)]):

        if i % 2 == 0 and i < 6:
            ax.set_title(r"$p = {}$".format(i//2), size=40, loc='right', x=1.20)

        if i % 3 == 0:
            ax.set_ylabel(r"$l = {}$".format(i//6), size=40)

        ax.set_yticks([])
        ax.set_xticks([])

        ax.imshow(im, cmap="RdBu")
        i += 1
    fig.savefig('GLM.png', bbox_inches='tight',
                  pad_inches=0, transparent=False)
    plt.show()
