import numpy as np
import sys

from itertools import product
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize
from utils import gau_kern, pad_with, zern_modes, circ_mask, crop_center
from scipy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift

def diffract(P,R,O,**kwargs):
    probe_size = kwargs['probe_size']
    rows = kwargs['rows']
    cols = kwargs['cols']
    px0 =  kwargs['px0']
    py0 = kwargs['py0']
    shift = kwargs['shift']

    O_dif = np.zeros((rows * cols, probe_size, probe_size),
            dtype=np.complex64)

    for k in range(rows*cols):
        x, y = R[k]
        x_loc = np.int(np.round((x - px0) / shift))
        y_loc = np.int(np.round((y - py0) / shift))
        O_illu = O[y:y+probe_size, x:x+probe_size]
        O_conv = O_illu * P
        O_dif[x_loc * rows + y_loc]= np.abs(fftshift(fft2(O_conv)))
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
    _O_gdif = None # saving checkpoint
    sse = np.zeros(iter)

    P_est = gau_kern(probe_size, probe_size / 2.35482,
            normalized=False)
    P_est = P_est > 0.5
    P_est = P_est.astype(np.int)

    idx = range(rows*cols)
    o = 0
    m = iter // n

    for k in range(iter):
        if permute:
            idx = np.random.permutation(idx)

        for u in idx:
            x, y = R[u]
            i = np.int(np.round((x - px0) / shift))
            j = np.int(np.round((y - py0) / shift))
            O_illu = O_est[y:y+probe_size, x:x+probe_size]
            O_illu_copy = np.copy(O_illu)
            O_g = O_illu * P_est
            O_gdif = fftshift(fft2(O_g))
            O_cdif = O_dif[i * rows + j] * np.exp(1j*np.angle(O_gdif))
            O_c = ifft2(ifftshift(O_cdif))
            O_est[y:y+probe_size, x:x+probe_size] = O_illu + (O_c - O_g) * alpha * \
                    np.conj(P_est) / np.max(np.abs(P_est))**2

            if k >= 2:
                P_est =  P_est + (O_c - O_g) * beta * \
                        np.conj(O_illu_copy) / np.max(np.abs(O_illu_copy))**2

            if i * rows + j == rows * cols // 2: # arbitrary
                _O_gdif = O_gdif

        e = np.abs(O_dif[rows * cols // 2])**2 - np.abs(_O_gdif)**2
        sse[k] = np.sum(np.sum(e**2 / pixels))

        if k % m == 0:
            O_n[o] = O_est
            o += 1

    def RMS(On):
        gamma = np.sum(O * np.conj(On)) / np.sum(np.abs(On)**2)
        return np.sum(np.abs(O - gamma*On)**2) / np.sum(np.abs(O)**2)

    Eo = np.array(list(map(RMS, O_n)))

    return O_est, P_est, Eo, sse


if __name__ == "__main__":
    import yaml
    import matplotlib.pyplot as plt
    from matplotlib import rc

    # Aesthetics
    plt.style.use('seaborn-notebook')
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

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

    phi = zern_modes([(2,6,1.0)], probe_size, w)
    P = gau_kern(w, probe_size / 2.35482,
            normalized=False)
    P = P > 0.5
    P = P.astype(np.int)

    Pphase = crop_center(np.abs(P*phi), probe_size, probe_size)
    Pphase /= np.max(Pphase)
    Pphase = np.clip(Pphase, 0.0, 1.0)
    P = crop_center(P, probe_size, probe_size)
    P = P * np.exp(1j * Pphase)

    n = 5
    O_dif = diffract(P, R, O, **params)
    O_dif_padded = np.array(list(map(lambda x : np.pad(x, (h - probe_size)//2, pad_with,
        padder=0), O_dif)))
    O_est, P_est, Eo, sse = epie(R, O, O_dif, n=n, permute=True, **params)


    fig0,ax0 = plt.subplots(3,2)
    ax0[0][0].imshow(np.abs(O_est), cmap='gray')
    ax0[0][1].imshow(np.angle(O_est), cmap='gray')
    ax0[1][0].imshow(np.abs(P_est), cmap='gray')
    ax0[1][1].imshow(np.angle(P_est), cmap='RdBu_r')
    ax0[2][0].imshow(np.abs(P), cmap='gray')
    ax0[2][1].imshow(np.angle(P), cmap='RdBu_r')


    fig1, ax1 = plt.subplots()
    ax1.plot(range(0, iter, iter // n), Eo, '--o', alpha=0.5)
    ax1.set_title('RMS error')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel(r'$E_o$')

    fig2, ax2 = plt.subplots()
    ax2.plot(sse[0:60] , '--o', alpha=0.5)
    ax2.set_title('Error Sum of Squares')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('SSE')


    plt.show()
