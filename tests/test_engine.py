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
    obj = amp * np.exp(j * phase)
    height, width = obj.shape

    ptycho = PytchoSimulator()

    beam = GLM(w0=.1, k=2, maxP=6, maxL=6)

    width = 80
    z = 1.0

    xx, yy = np.meshgrid(np.mgrid[-10:10:width*j], np.mgrid[-10:10:width*j]);

    # Calculate the cylindrical coordinates
    r = np.sqrt(xx**2 + yy**2);
    phi = np.arctan2(yy, xx);

    illu_func = beam.field(r, phi, z, p=0, l=3)

    '''
    phi = zernike_modes([(0, 6, 10)], ptycho.probe, width)
    gau_mask = gau_kern(width, ptycho.probe / 2.35482,
                        normalize=False)
    gau_mask = gau_mask > 0.5
    gau_mask = gau_mask.astype(np.int)

    illu_func_phase = crop_center(np.abs(gau_mask*phi), ptycho.probe, ptycho.probe)
    illu_func_pamp = crop_center(gau_mask, ptycho.probe, ptycho.probe)
    illu_func_pamp = illu_func_pamp / illu_func_pamp.max()
    illu_func_phase = illu_func_phase / illu_func_phase.max()
    illu_func = illu_func_pamp * np.exp(1j * illu_func_phase)
    '''

    err_ival = 2
    diff_patterns = ptycho.diffract(obj, illu_func, mode='random', amount=0.5)
    obj_est, rms, sse = ptycho.pie(obj, illu_func, diff_patterns,
                                                   err_ival=err_ival,
                                                   permute=True)

    x_sse = range(0, ptycho.iterations, err_ival)
    y_sse = sse
    x_Eo = range(0, ptycho.iterations, err_ival)
    y_Eo = rms

    fig, ax = plt.subplots(2, 2)

    ax[0][0].imshow(np.abs(obj_est), cmap='gray')
    ax[0][0].set_title('Estimated Object\'s amplitude')
    ax[0][1].imshow(np.angle(obj_est), cmap='gray', vmin=-pi, vmax=pi)
    ax[0][1].set_title('Estimated Object\'s phase')

    ax[1][0].imshow(np.abs(illu_func), cmap='gray')
    ax[1][0].set_title('Actual Probe\'so amplitude')
    ax[1][1].imshow(np.angle(illu_func), cmap='gray')
    ax[1][1].set_title('Actual Probe\'so phase')

    figg, axx = plt.subplots()

    axx.plot(x_Eo, y_Eo, lw=1)
    axx.set_title(r'RMS Error for PIE')
    axx.set_xlabel('Iterations')
    axx.set_ylabel('En')

    plt.show()
