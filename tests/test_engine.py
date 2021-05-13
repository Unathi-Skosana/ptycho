if __name__ == "__main__":
    import yaml
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np

    from skimage.exposure import rescale_intensity
    from skimage.restoration import unwrap_phase
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

    ptycho = PytchoSimulator(alpha=1.0, beta=1.0, probe=70,
                             start=(20, 20), shift=5, rc=(30, 30),
                             iterations=30)
    width = 70
    zz = 100
    xx, yy = np.meshgrid(np.mgrid[-5:5:width*j], np.mgrid[-5:5:width*j])

    # Calculate the cylindrical coordinates
    rr = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)

    GL_beam = GLM(w0=70, k=20, maxL=6, maxP=6)
    GH_beam = GHM(w0=70, k=20, maxL=6, maxM=6)

    GL_c = np.zeros(GL_beam.shape)
    GL_c[2, 2] = 1.0

    GH_c = np.zeros(GH_beam.shape)
    GH_c[5, 3] = 1.0
    GH_c[3, 5] = 1.0

    GH_illu_func = GH_beam.field(xx, yy, zz, GH_c)
    GL_illu_func = GL_beam.field(rr, phi, zz, GL_c)

    gau_mask = gau_kern(width, ptycho.probe / np.sqrt(8 * np.log(2)),
                        normalize=False)
    gau_mask = gau_mask > 0.5
    gau_mask = gau_mask.astype(np.int64)

    GL_illu_func = (np.abs(GH_illu_func) * gau_mask) * np.exp(1j *
            (rescale_intensity(np.angle(GH_illu_func), out_range=(0, 1.0)) * gau_mask))

    phi = zernike_modes([(0, 6, 1.0)], ptycho.probe, width)
    illu_func = gau_mask * np.exp(j * np.abs(phi))

    diff_patterns = ptycho.diffract(obj, illu_func)
    err_ival = 1
    obj_est, illu_est, rms, sse = ptycho.epie(obj, diff_patterns,
            err_ival=err_ival, hold=5, permute=True)

    x_sse = range(0, ptycho.iterations, err_ival)
    y_sse = sse
    x_Eo = range(0, ptycho.iterations, err_ival)
    y_Eo = rms

    fig0, ax0 = plt.subplots()
    ax0.imshow(rescale_intensity(np.abs(illu_est), out_range=(0.0, 1.0)),
            cmap='gray', vmin=0.0, vmax=1.0)
    ax0.set_axis_off()

    fig1, ax1 = plt.subplots()
    cax1 = ax1.imshow(rescale_intensity(np.angle(illu_est), out_range=(-np.pi, np.pi)),
            cmap='gray', vmin=-np.pi, vmax=np.pi)
    cbar = fig1.colorbar(cax1, ticks=[-np.pi, 0, np.pi], shrink=0.7)
    cbar.ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax1.set_axis_off()

    fig2, ax2 = plt.subplots()
    ax2.imshow(rescale_intensity(np.abs(obj_est), out_range=(0.0, 1.0)),
            cmap='gray', vmin=0.0, vmax=1.0)
    ax2.set_axis_off()

    fig3, ax3 = plt.subplots()
    cax3 = ax3.imshow(rescale_intensity(np.angle(obj_est), out_range=(-np.pi, np.pi)),
            cmap='gray', vmin=-np.pi, vmax=np.pi)
    cbar = fig3.colorbar(cax3, ticks=[-np.pi, 0, np.pi], shrink=0.7)
    cbar.ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax3.set_axis_off()

    import re
    from ast import literal_eval

    y0 = """[0.90483692 0.90374113 0.90429303 0.8973554  0.89734811 0.70318158
 0.37066099 0.28816683 0.27800489 0.27600822 0.27553632 0.27490912
 0.27440053 0.27458025 0.27395239 0.27408412 0.27416223 0.27446098
 0.2737171  0.27364155 0.27379874 0.27363329 0.27357349 0.27331187
 0.27365687 0.27328793 0.27347618 0.27345029 0.27352441 0.27340579]"""
    y0 = re.sub(r"([^[])\s+([^]])", r"\1, \2", y0)
    y0 = np.array(literal_eval(y0))


    y1 = """[0.95244364 0.94841735 0.94715397 0.92863745 0.93502776 0.79921412
 0.61188945 0.53625176 0.51568352 0.51037334 0.50919995 0.50875299
 0.50747153 0.50706343 0.50669917 0.50675973 0.50764833 0.50782289
 0.50805125 0.50837725 0.50690982 0.50666131 0.50641156 0.50656684
 0.50683789 0.50630424 0.50639292 0.5063244  0.50712751 0.50694192]"""
    y1 = re.sub(r"([^[])\s+([^]])", r"\1, \2", y1)
    y1 = np.array(literal_eval(y1))

    y2 = """[0.9647911  0.95570849 0.94920891 0.93548106 0.94563617 0.83348097
 0.61704139 0.59381459 0.59012862 0.58843936 0.58818075 0.58725761
 0.58807764 0.58761439 0.58757392 0.58803968 0.58804852 0.58820154
 0.58846785 0.58837414 0.58869972 0.58858621 0.58892867 0.58910458
 0.58904577 0.58895214 0.58934404 0.58957106 0.58978409 0.58941013]"""
    y2 = re.sub(r"([^[])\s+([^]])", r"\1, \2", y2)
    y2 = np.array(literal_eval(y2))

    y3 = """[0.9363062  0.93352291 0.90524763 0.89630739 0.89689763 0.72520037
 0.48846605 0.45786906 0.45612987 0.45576941 0.45569499 0.45568492
 0.45569623 0.4557357  0.45573858 0.45575452 0.45576305 0.45576819
 0.45577219 0.45577402 0.4557767  0.45577632 0.45577741 0.45577811
 0.45577836 0.45577907 0.45577938 0.45577957 0.45577966 0.4557798]"""
    y3 = re.sub(r"([^[])\s+([^]])", r"\1, \2", y3)
    y3 = np.array(literal_eval(y3))

    fig4, ax4 = plt.subplots()
    ax4.plot(x_Eo, y0, 'C1o:', mec="1.0")
    ax4.plot(x_Eo, y1, 'C2o:', mec="1.0")
    ax4.plot(x_Eo, y2, 'C3o:', mec="1.0")
    ax4.plot(x_Eo, y3, 'C4o:', mec="1.0")
    ax4.set_xticks([0 ,5, 10, 15, 20, 25, 30])

    ax4.set_title('Root mean square error')
    ax4.set_xlabel(r'$n$')
    ax4.set_ylabel(r'$E_O(n)$')


    fig0.savefig('epie_probe_amplitude.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig1.savefig('epie_probe_phase.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig2.savefig('epie_obj_amplitude.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig3.savefig('epie_obj_phase.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig4.savefig('rms.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    plt.show()
