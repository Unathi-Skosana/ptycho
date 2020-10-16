import numpy as np
import csv
import sys

from joblib import Parallel, delayed
from utils import gau_kern, pad_with, zern_modes, circ_mask, crop_center
from scipy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift
from numpy import pi
from itertools import chain

def pertube_diff(d,v):
    def f(a):
        c = list(map(lambda I : I + np.random.uniform(low=-v/100 * I, high=v/100 * I), a))
        return np.array(c)

    b = np.fromiter(chain.from_iterable(f(i) for i in d), dtype=d.dtype)
    return np.reshape(b, d.shape)

def diffract(P,R,O,**kwargs):
    probe_size = kwargs['probe_size']
    rows = kwargs['rows']
    cols = kwargs['cols']
    px0 =  kwargs['px0']
    py0 = kwargs['py0']
    shift = kwargs['shift']

    O_dif = np.zeros((rows * cols, probe_size, probe_size),
            dtype=np.float64)

    def d(k):
        x, y = R[k]
        x_loc = np.int(np.round((x  - px0) / shift))
        y_loc = np.int(np.round((y  - py0) / shift))
        O_illu = O[y:y+probe_size, x:x+probe_size]
        O_conv = O_illu * P
        diff = np.abs(fftshift(fft2(O_conv)))
        O_dif[x_loc * rows + y_loc] = diff

    Parallel(n_jobs=8, prefer="threads")(delayed(d)(k) for k in range(rows*cols))
    return O_dif


def epie(R,O,O_dif,n=10,permute=False,**kwargs):
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    iter = kwargs['iter']
    rows = kwargs['rows']
    cols = kwargs['cols']
    probe_size = kwargs['probe_size']
    px0 =  kwargs['px0']
    py0 = kwargs['py0']
    shift = kwargs['shift']

    h,w = O.shape
    pixels = h * w
    O_est = np.zeros((h, w), dtype=np.complex128)
    O_n = np.zeros((n, h, w), dtype=np.complex128)
    sse = np.zeros(iter)
    _O_gdif = None # saving checkpoint

    P_est = gau_kern(probe_size, probe_size / 2.35482,
            normalized=False)
    P_est = P_est > 0.5
    P_est = P_est.astype(np.int)

    idx = range(rows * cols)
    m = iter // n
    o = 0
    k = 0

    while k < iter:
        if permute:
            idx = np.random.permutation(idx)

        for u in idx:
            x, y = R[u]

            O_illu = O_est[y:y+probe_size, x:x+probe_size]
            O_illu_copy = np.copy(O_illu)
            O_g = O_illu * P_est
            O_gdif = fftshift(fft2(O_g))

            #if u == 0 or u == 4 or u == 81 or u == 101:
            #    x += np.random.randint(low=-2, high=2)
            #    y += np.random.randint(low=-2, high=2)

            i = np.int(np.round((x - px0) / shift))
            j = np.int(np.round((y - py0) / shift))

            #O_cdif = O_dif[i * rows + j] * np.exp(1j * np.angle(O_gdif))
            
            #O_cdif = pertube_diff(O_dif[i*rows + j] / O_dif[i*rows + j].max(),
            #    50) * np.exp(1j * np.angle(O_gdif))

            O_cdif = np.random.poisson(6.24 * 10e9 * O_dif[i * rows + j] / O_dif[i*rows
                +j].max()) / (6.24 * 10e9) * np.exp(1j * np.angle(O_gdif))

            O_c = ifft2(ifftshift(O_cdif))
            O_est[y:y+probe_size, x:x+probe_size] = O_illu + (O_c - O_g) * alpha * \
                    np.conj(P_est) / np.max(np.abs(P_est))**2

            if k >= 1:
                P_est =  P_est + (O_c - O_g) * beta * \
                    np.conj(O_illu_copy) / np.max(np.abs(O_illu_copy))**2

            if i * rows + j == rows * cols // 2: # arbitrary
                _O_gdif = O_gdif

        e = (np.abs(O_dif[rows * cols // 2])**2 - np.abs(_O_gdif)**2)**2
        sse[k] = np.abs(np.sum(np.sum(e / pixels)))

        if k % m == 0:
            O_n[o] = O_est
            o += 1
        k += 1

    def RMS(On):
        gamma = np.sum(O * np.conj(On)) / np.sum(np.abs(On)**2)
        return np.sum(np.abs(O - gamma*On)**2) / np.sum(np.abs(O)**2)

    Eo = np.array(list(map(RMS, O_n)))

    return O_est, P_est, Eo, sse


if __name__ == "__main__":

    import yaml
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from skimage.restoration import unwrap_phase
    from skimage.io import imread, imsave
    from skimage import img_as_float, img_as_float64
    from itertools import product

    # Aesthetics
    plt.style.use('tint')

    fn = sys.argv[1]
    im1_fn = sys.argv[2]
    im2_fn = sys.argv[3]

    params = None
    with open(fn) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    im1 = img_as_float(imread(im1_fn, as_gray=True))
    im2 = img_as_float(imread(im2_fn, as_gray=True))
    amp = im1
    phase = im2
    O = amp * np.exp(1j * phase)
    h,w = O.shape

    iter = params['iter']
    probe_size = params['probe_size']
    rows, cols = params['rows'], params['cols']
    shift = params['shift']
    px0, py0 = params['px0'], params['py0']

    px = np.arange(px0, px0 + rows*shift, shift)
    py = np.arange(py0, py0 + cols*shift, shift)
    R =  np.array(list(product(px, py)))

    phi = zern_modes([(-2,4,1.0)], probe_size, w)
    gau_mask = gau_kern(w, probe_size / 2.35482,
            normalized=False)
    gau_mask = gau_mask > 0.5
    gau_mask = gau_mask.astype(np.int)

    pphase = crop_center(np.abs(gau_mask*phi), probe_size,
        probe_size)
    pamp = crop_center(gau_mask, probe_size, probe_size)

    pamp =  pamp / pamp.max()
    pphase = pphase / pphase.max()

    P = pamp * np.exp(1j * pphase)

    n = 10
    O_dif = diffract(P, R, O, **params)
    O_dif_padded = np.array(list(map(lambda x : np.pad(x, (h - probe_size)//2, pad_with,
        padder=0), O_dif)))
    O_est, P_est, Eo, sse = epie(R, O, O_dif, n=n, permute=True, **params)

    x_sse = range(0, iter)
    y_sse =  sse

    x_Eo = range(0, iter, iter // n)
    y_Eo = Eo

    fig, ax = plt.subplots(3,2)

    ax[0][0].imshow(np.abs(O_est), cmap='gray')
    ax[0][0].set_title('Estimated Object\'s amplitude')
    ax[0][1].imshow(np.angle(O_est), cmap='gray')
    ax[0][1].set_title('Estimated Object\'s phase')

    ax[1][0].imshow(np.abs(P_est), cmap='gray')
    ax[1][0].set_title('Estimated Probe\'s amplitude')
    ax[1][1].imshow(np.abs(P_est) * np.angle(P_est), cmap='gray')
    ax[1][1].set_title('Estimated Probe\'s phase')

    ax[2][0].imshow(np.abs(P), cmap='gray')
    ax[2][0].set_title('Actual Probe\'s amplitude')
    ax[2][1].imshow(np.angle(P), cmap='gray')
    ax[2][1].set_title('Actual Probe\'s phase')

    figg, axx = plt.subplots()

    axx.plot(x_Eo, y_Eo, alpha=0.5, lw=1)
    axx.set_title(r'RMS Error for ePIE')
    axx.set_xlabel('Iterations')
    axx.set_ylabel('En')

    plt.show()

