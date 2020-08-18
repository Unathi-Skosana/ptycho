import yaml
import numpy as np
import sys
import matplotlib.pyplot as plt

from itertools import product
from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize
from utils import gau_kern
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


def pie(P,R,O_dims,O_dif, permute=False, **kwargs):
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    iter = kwargs['iter']
    rows = kwargs['rows']
    cols = kwargs['cols']
    probe_size = kwargs['probe_size']
    px0 =  kwargs['px0']
    py0 = kwargs['py0']
    shift = kwargs['shift']

    wh = O_dims[0] * O_dims[1]
    O = np.zeros(O_dims, dtype=np.complex128)
    P_c = np.zeros((probe_size, probe_size), dtype=np.complex128)
    sse = np.zeros(iter)
    _O_gdif = None # saving checkpoint

    idx = range(rows*cols)

    if permute:
        idx = np.random.permutation(idx)

    for k in range(iter):
        for u in idx:
            x, y = R[u]
            i = np.int(np.round((x - px0) / shift))
            j = np.int(np.round((y - py0) / shift))
            O_illu = O[y:y+probe_size, x:x+probe_size]
            O_g = O_illu * P
            O_gdif = fftshift(fft2(O_g))
            O_cdif = O_dif[i * rows + j] * np.exp(1j*np.angle(O_gdif))
            O_c = ifft2(ifftshift(O_cdif))
            O[y:y+probe_size, x:x+probe_size] = O_illu + (O_c - O_g) * alpha * \
                    np.conj(P) / np.max(np.abs(P))**2
            P_c =  P_c + (O_c - O_g) * beta * \
                    np.conj(O_illu) / np.max(np.abs(O_illu))**2

            if i * rows + j == rows * cols // 2: # arbitrary
                _O_gdif = O_gdif

        e = np.abs(O_dif[rows * cols // 2])**2 - np.abs(_O_gdif)**2
        sse[k] = np.sum(np.sum(e**2 / wh))

    return O, P_c, sse


if __name__ == "__main__":
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
    obj = amp * np.exp(1j * phase)

    probe_size = params['probe_size']
    rows, cols = params['rows'], params['cols']
    shift = params['shift']
    px0, py0 = params['px0'], params['py0']

    px = np.arange(px0, px0 + rows*shift, shift)
    py = np.arange(py0, py0 + cols*shift, shift)
    R =  np.array(list(product(px, py)))

    probe = gau_kern(probe_size, probe_size / 2.35482,
            normalized=False)
    probe = probe > 0.5
    probe = probe.astype(np.int)

    obj_diff = diffract(probe, R, obj, **params)
    a, b, sse = pie(probe, R, obj.shape, obj_diff, permute=True, **params)

    fig,ax = plt.subplots(1,4)
    ax[0].imshow(np.abs(a), cmap='gray')
    ax[1].imshow(np.angle(a), cmap='gray')
    ax[2].imshow(np.abs(b), cmap='gray')
    ax[3].imshow(np.angle(b), cmap='gray')

    plt.show()
