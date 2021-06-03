if __name__ == "__main__":
    import yaml
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np

    from skimage.exposure import rescale_intensity
    from skimage.restoration import unwrap_phase
    from skimage.util import invert
    from skimage.io import imread
    from skimage import img_as_float

    from common.constants import j, pi
    from utils.filters import gau_kern, circ_mask
    from utils.formatting import crop_center
    from utils.aberrations import zernike_modes

    from engine.simulator import PytchoSimulator
    from beams.GaussianLaguerreBeams import GaussLaguerreModeSet as GLM
    from beams.GaussianHermiteBeams import GaussHermiteModeSet as GHM

    plt.style.use('mint')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-a", "--amplitude",
                        default="images/cameraman.tif",
                        help="Amplitude image")
    parser.add_argument("-p", "--phase",
                        default="images/lena.tif",
                        help="Phase image")
    ARGS = parser.parse_args()

    amp = img_as_float(imread(ARGS.amplitude, as_gray=True))
    phase = np.angle(np.exp(1j * img_as_float(imread(ARGS.phase,
        as_gray=True))))
    prob_phase = img_as_float(imread('images/spiral.tif', as_gray=True))
    obj = amp * np.exp(j * phase)
    height, width = obj.shape

    ptycho = PytchoSimulator(alpha=0.05, beta=1.0, probe=70,
                             start=(20, 20), shift=5, rc=(30, 30),
                             iterations=30)
    width = 70
    zz = 100
    xx, yy = np.meshgrid(np.mgrid[-35:35:width*j], np.mgrid[-35:35:width*j])

    # Calculate the cylindrical coordinates
    rr = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)

    GL_beam = GLM(w0=0.35, k=20, maxL=6, maxP=6)
    GH_beam = GHM(w0=70, k=20, maxL=6, maxM=6)

    GL_c = np.zeros(GL_beam.shape)
    GL_c[1, 2] = 1.0

    GH_c = np.zeros(GH_beam.shape)
    GH_c[1, 3] = 1.0

    GH_illu_func = GH_beam.field(xx, yy, zz, GH_c)
    GL_illu_func = GL_beam.field(rr, phi, zz, GL_c)

    cmask = circ_mask(70, (70//2,70//2), 35, 1.0, inverse=False)

    GL_illu_func = (np.abs(GL_illu_func) * cmask) * np.exp(1j *
            (rescale_intensity(np.angle(GL_illu_func), out_range=(0, 1.0)) * cmask))

    phi = zernike_modes([(0, 6, 1.0)], ptycho.probe, width)
    illu_func = cmask * np.exp(j * np.abs(phi))

    diff_patterns = ptycho.diffract(obj, GL_illu_func)
    err_ival = 1
    obj_est, illu_func_est, RMS, R_factor = ptycho.repie(obj, diff_patterns,
            err_ival=err_ival, hold=5, permute=True)

    fig0, ax0 = plt.subplots()
    ax0.imshow(rescale_intensity(np.abs(GL_illu_func), out_range=(0.0, 1.0)),
            cmap='gray', vmin=0.0, vmax=1.0)
    ax0.set_axis_off()

    fig1, ax1 = plt.subplots()
    cax1 = ax1.imshow(rescale_intensity(np.angle(obj_est), out_range=(-np.pi, np.pi)),
            cmap='gray', vmin=-np.pi, vmax=np.pi)
    cbar = fig1.colorbar(cax1, ticks=[-np.pi, 0, np.pi], shrink=0.7)
    cbar.ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax1.set_axis_off()

    fig2, ax2 = plt.subplots()
    x = range(0, ptycho.iterations, err_ival)
    ax2.plot(x, RMS, 'C1o:', mec="1.0")

    fig3, ax3 = plt.subplots()
    x = range(0, ptycho.iterations, err_ival)
    ax3.plot(x, R_factor, 'C2o:', mec="1.0")

    fig4, ax4 = plt.subplots()
    ax4.imshow(rescale_intensity(np.abs(illu_func_est), out_range=(0.0, 1.0)),
            cmap='gray', vmin=0.0, vmax=1.0)
    ax4.set_axis_off()

    fig5, ax5 = plt.subplots()
    cax5 = ax5.imshow(rescale_intensity(np.angle(illu_func_est), out_range=(-np.pi, np.pi)),
            cmap='gray', vmin=-np.pi, vmax=np.pi)
    cbar = fig5.colorbar(cax5, ticks=[-np.pi, 0, np.pi], shrink=0.7)
    cbar.ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax5.set_axis_off()

    # fig6, ax6 = plt.subplots()
    # ax6.imshow(rescale_intensity(np.abs(diff_patterns[11]), out_range=(0.0, 1.0)), cmap='RdBu', vmin=0.0, vmax=1.0)
    # ax6.set_axis_off()

    # fig7, ax7 = plt.subplots()
    # ax7.imshow(rescale_intensity(np.abs(diff_patterns[12]), out_range=(0.0, 1.0)), cmap='RdBu', vmin=0.0, vmax=1.0)
    # ax7.set_axis_off()

    # fig8, ax8 = plt.subplots()
    # ax8.imshow(rescale_intensity(np.abs(diff_patterns[13]), out_range=(0.0, 1.0)), cmap='RdBu', vmin=0.0, vmax=1.0)
    # ax8.set_axis_off()

    fig0.savefig('epie_obj_amplitude.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig1.savefig('epie_obj_phase.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig4.savefig('epie_probe_amplitude.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig5.savefig('epie_probe_phase.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    # fig4.savefig('rms.png', bbox_inches='tight',
    #              pad_inches=0, transparent=False)
    # fig5.savefig('diff1.png', bbox_inches='tight',
    #              pad_inches=0, transparent=False)
    # fig6.savefig('diff2.png', bbox_inches='tight',
    #              pad_inches=0, transparent=False)
    # fig7.savefig('diff3.png', bbox_inches='tight',
    #              pad_inches=0, transparent=False)
    # fig8.savefig('diff4.png', bbox_inches='tight',
    #              pad_inches=0, transparent=False)
    # fig9.savefig('R_factor.png', bbox_inches='tight',
    #             pad_inches=0, transparent=False)
    plt.show()
