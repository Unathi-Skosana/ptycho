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

    ptycho = PytchoSimulator(alpha=0.25, beta=1.0, probe=70,
                             start=(20, 20), shift=5, rc=(30, 30),
                             iterations=1)

    wavelength = 624 * 1e-9
    k = 2 * np.pi / wavelength
    w0 = 20 * 1e-6

    N = 70
    L = 100 * 1e-6
    x0 = np.linspace(-L/2, L/2, N)
    y0 = np.linspace(-L/2, L/2, N)
    xv, yv = np.meshgrid(x0, y0)
    zz = 1*1e-9

    beam = GLM(w0=w0, k=k, maxP=8, maxL=8)
    rr = np.sqrt(xv**2 + yv**2)
    phi = np.arctan2(yv, xv)

    cmask = circ_mask(N, (N//2, N//2), N//2, 1.0, inverse=False)

    c = np.zeros(beam.shape)
    c[2, 2] = 1.0
    u1 = beam.field(rr, phi, zz, c)
    u1 = (np.abs(u1) * cmask) * np.exp(1j *
            (rescale_intensity(np.angle(u1) * cmask, out_range=(0, 1.0))))

    diff_patterns = ptycho.diffract(obj, u1) 
    err_ival = 1

    obj_est, illu_func_est, RMS, R_factor = ptycho.repie(obj, diff_patterns,
            err_ival=err_ival, hold=5, permute=True)

    fig0, ax0 = plt.subplots()
    ax0.imshow(rescale_intensity(np.abs(obj_est), out_range=(0.0, 1.0)),
            cmap='gray', vmin=0.0, vmax=1.0)
    ax0.set_axis_off()

    fig1, ax1 = plt.subplots()
    cax1 = ax1.imshow(rescale_intensity(unwrap_phase(np.angle(obj_est)), out_range=(-np.pi, np.pi)),
            cmap='gray', vmin=-np.pi, vmax=np.pi)
    ax1.set_axis_off()

    fig2, ax2 = plt.subplots()
    x = range(0, ptycho.iterations, err_ival)
    ax2.plot(x, RMS, 'C1o:', mec="1.0")
    ax2.set_title("RMS")

    fig3, ax3 = plt.subplots()
    x = range(0, ptycho.iterations, err_ival)
    ax3.plot(x, R_factor, 'C2o:', mec="1.0")
    ax3.set_title("R-factor")

    fig4, ax4 = plt.subplots()
    ax4.imshow(rescale_intensity(np.abs(illu_func_est), out_range=(0.0, 1.0)),
            cmap='gray', vmin=0.0, vmax=1.0)
    ax4.set_axis_off()

    fig5, ax5 = plt.subplots()
    cax5 = ax5.imshow(rescale_intensity(np.angle(illu_func_est), out_range=(-np.pi, np.pi)),
            cmap='gray', vmin=-np.pi, vmax=np.pi)
    ax5.set_axis_off()

    fig6, ax6 = plt.subplots()
    ax6.imshow(rescale_intensity(np.abs(u1), out_range=(0.0, 1.0)),
            cmap='gray', vmin=0.0, vmax=1.0)
    ax6.set_axis_off()

    fig7, ax7 = plt.subplots()
    cax7 = ax7.imshow(rescale_intensity(unwrap_phase(np.angle(u1)), out_range=(-np.pi, np.pi)),
            cmap='gray', vmin=-np.pi, vmax=np.pi)
    ax7.set_axis_off()

    fig0.savefig('v30_obj_ampl.png')
    fig1.savefig('v30_obj_phase.png')
    fig4.savefig('v30_probe_ampl.png')
    fig5.savefig('v30_probe_phase.png')

    plt.show()
