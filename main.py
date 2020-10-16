if __name__ == "__main__": 
    import yaml
    import matplotlib.pyplot as plt
    import numpy as np
    import sys

    from skimage.restoration import unwrap_phase
    from skimage.io import imread, imsave
    from skimage import img_as_float
    from itertools import product
    from utils import gau_kern, zern_modes, crop_center
    from epie import diffract, simulate

    plt.style.use('mint')

    fn = sys.argv[1]
    im1_fn = sys.argv[2]
    im2_fn = sys.argv[3]

    amp = img_as_float(imread(im1_fn, as_gray=True))
    phase = img_as_float(imread(im2_fn, as_gray=True))

    params = None
    with open(fn) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    N = params['N']
    probe = params['probe']
    r, c = params['r'], params['c']
    shift = params['shift']
    px, py = params['px'], params['py']

    obj = amp * np.exp(1j * phase)
    h, w = obj.shape

    x = np.arange(px, px + r * shift, shift)
    y = np.arange(py, py + c * shift, shift)
    R = np.array(list(product(x, y)))

    phi = zern_modes([(-1, 1, 10)], probe, w)

    gau_mask = gau_kern(w, probe / 2.35482,
                        normalized=False)
    gau_mask = gau_mask > 0.5
    gau_mask = gau_mask.astype(np.int)

    illu_func_phase = crop_center(np.abs(gau_mask*phi), probe, probe)
    illu_func_pamp = crop_center(gau_mask, probe, probe)
    illu_func_pamp = illu_func_pamp / illu_func_pamp.max()
    illu_func_phase = illu_func_phase / illu_func_phase.max()
    illu_func = illu_func_pamp * np.exp(1j * illu_func_phase)

    n = 100
    diff_patterns = diffract(obj, illu_func, R, **params)
    obj_est, illu_func_est, rms, sse = simulate(obj, R,
                                                diff_patterns,
                                                hold=10,
                                                rms_n=n,
                                                permute=True,
                                                **params)

    x_sse = range(0, N)
    y_sse = sse
    x_Eo = range(0, N, N // n)
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

    axx.plot(x_Eo, y_Eo, alpha=0.5, lw=1)
    axx.set_title(r'RMS Error for PIE')
    axx.set_xlabel('Iterations')
    axx.set_ylabel('E_n')

    plt.show()
