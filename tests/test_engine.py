if __name__ == "__main__":
    import yaml
    import matplotlib.pyplot as plt
    import numpy as np
    import sys

    from skimage.restoration import unwrap_phase
    from skimage.io import imread
    from skimage import img_as_float
    from itertools import product

    from utils.filters import gau_kern
    from utils.formatting import crop_center
    from utils.aberrations import zernike_poly
    from engine.simulator import diffract, epie

    plt.style.use('mint')

    fn = sys.argv[1]
    im1_fn = sys.argv[2]
    im2_fn = sys.argv[3]

    amp = img_as_float(imread(im1_fn, as_gray=True))
    phase = img_as_float(imread(im2_fn, as_gray=True))

    height, width = phase.shape

    params = None
    with open(fn) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    iterations = params['iterations']
    probe = params['probe']
    rows, cols = params['rows'], params['cols']
    shift = params['shift']
    p_x, p_y = params['p_x'], params['p_y']

    obj = amp * np.exp(1j * phase)

    x = np.arange(p_x, p_x + rows * shift, shift)
    y = np.arange(p_y, p_y + cols * shift, shift)
    R = np.array(list(product(x, y)))

    phi = zernike_poly([(0, 0, 2)], probe, width)

    gau_mask = gau_kern(width, probe / 2.35482,
                        normalize=False)
    gau_mask = gau_mask > 0.5
    gau_mask = gau_mask.astype(np.int)

    illu_func_phase = crop_center(np.abs(gau_mask*phi), probe, probe)
    illu_func_pamp = crop_center(gau_mask, probe, probe)
    illu_func_pamp = illu_func_pamp / illu_func_pamp.max()
    illu_func_phase = illu_func_phase / illu_func_phase.max()
    illu_func = illu_func_pamp * np.exp(1j * illu_func_phase)

    rms_n = 100
    diff_patterns = diffract(obj, illu_func,
                             R, mode='position',
                             **params)
    obj_est, illu_func_est, rms, sse = epie(obj, R,
                                            diff_patterns,
                                            hold=10,
                                            rms_n=rms_n,
                                            permute=True,
                                            **params)

    x_sse = range(0, iterations)
    y_sse = sse
    x_Eo = range(0, iterations, iterations // rms_n)
    y_Eo = rms

    fig, ax = plt.subplots(3, 2)

    ax[0][0].imshow(np.abs(obj_est), cmap='gray')
    ax[0][0].set_title('Estimated Object\'s amplitude')
    ax[0][1].imshow(np.angle(obj_est), cmap='gray')
    ax[0][1].set_title('Estimated Object\'s phase')

    ax[1][0].imshow(np.abs(illu_func_est), cmap='gray')
    ax[1][0].set_title('Estimated Probe\'s amplitude')
    ax[1][1].imshow(unwrap_phase(np.angle(illu_func_est)), cmap='gray')
    ax[1][1].set_title('Estimated Probe\'so phase')

    ax[2][0].imshow(np.abs(illu_func), cmap='gray')
    ax[2][0].set_title('Actual Probe\'so amplitude')
    ax[2][1].imshow(np.angle(illu_func), cmap='gray')
    ax[2][1].set_title('Actual Probe\'so phase')

    figg, axx = plt.subplots()

    axx.plot(x_Eo, y_Eo, lw=1)
    axx.set_title(r'RMS Error for PIE')
    axx.set_xlabel('Iterations')
    axx.set_ylabel('En')

    plt.show()
