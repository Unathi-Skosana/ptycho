if __name__ == "__main__":
    import yaml
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np

    from skimage.restoration import unwrap_phase
    from skimage.io import imread
    from skimage import img_as_float

    from common.constants import j, pi
    from utils.filters import gau_kern
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
    phase = img_as_float(imread(ARGS.phase, as_gray=True))
    prob_phase = img_as_float(imread('images/spiral.tif', as_gray=True))
    obj = amp * np.exp(j * phase)
    height, width = obj.shape

    ptycho = PytchoSimulator(alpha=1.0, beta=1.0, probe=70,
                             start=(2, 2), shift=20, rc=(10, 10),
                             iterations=500)

    beam = GHM(w0=70, k=20, maxL=6, maxM=6)
    width = 70
    zz = 0.5
    xx, yy = np.meshgrid(np.mgrid[-5:5:width*j], np.mgrid[-5:5:width*j]);

    # Calculate the cylindrical coordinates
    rr = np.sqrt(xx**2 + yy**2);
    phi = np.arctan2(yy, xx);

    c = np.zeros(beam.shape)
    c[5, 3] = 1.0
    c[3, 5] = 1.0

    illu_func = beam.field(xx, yy, zz, c)

    '''
    phi = zernike_modes([(-2, 6, 1.0)], ptycho.probe, width)
    gau_mask = gau_kern(width, ptycho.probe / np.sqrt(8 * np.log(2)),
                        normalize=False)
    gau_mask = gau_mask > 0.5
    gau_mask = gau_mask.astype(np.int)

    illu_func_phase = crop_center(np.abs(prob_phase), ptycho.probe, ptycho.probe)
    illu_func_pamp = crop_center(gau_mask, ptycho.probe, ptycho.probe)
    illu_func = illu_func_pamp * np.exp(j * illu_func_phase)
    '''

    err_ival = 1
    diff_patterns = ptycho.diffract(obj, illu_func)
    obj_est, rms, sse = ptycho.rpie(obj, illu_func, diff_patterns,
                                   err_ival=err_ival, permute=True)

    x_sse = range(0, ptycho.iterations, err_ival)
    y_sse = sse
    x_Eo = range(0, ptycho.iterations, err_ival)
    y_Eo = rms

    fig1, axes1 = plt.subplots(1, 2)
    ax1, ax2 = axes1.ravel()
    ax1.imshow(np.abs(obj_est), cmap='gray')
    ax1.set_axis_off()
    ax2.imshow(np.angle(obj_est), cmap='gray')
    ax2.set_axis_off()

    fig2, ax3 = plt.subplots()
    ax3.plot(x_Eo, y_Eo)
    ax3.set_title('Root mean square error')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Error')

    fig1.savefig('rpie_gh.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
    fig2.savefig('rpie_gh_rms.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)

    plt.show()
