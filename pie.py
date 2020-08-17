import yaml
import numpy as np
import sys
import matplotlib.pyplot as plt

from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize
from utils import gau_kern
from scipy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift

def diffract(probe,px,py,obj,**kwargs):
    probe_size = kwargs['probe_size']
    rows = kwargs['rows']
    cols = kwargs['cols']
    px0 =  kwargs['px0']
    py0 = kwargs['py0']
    shift = kwargs['shift']
    obj_diff = np.zeros((rows, cols, probe_size, probe_size),
            dtype=np.complex64)

    for x in px:
        for y in py:
            x_loc = np.int(np.round((x - px0) / shift))
            y_loc = np.int(np.round((y - py0) / shift))
            illu_obj = obj[y:y+probe_size, x:x+probe_size]
            obj_conv = illu_obj * probe
            obj_diff[y_loc, x_loc]= np.abs(fftshift(fft2(obj_conv)))
    return obj_diff



def pie(probe,px,py,obj_diff,**kwargs):
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    iter = kwargs['iter']
    probe_size = kwargs['probe_size']
    px0 =  kwargs['px0']
    py0 = kwargs['py0']
    shift = kwargs['shift']
    obj = np.zeros((256,256), dtype=np.complex128)

    for i in range(iter):
        for x in px:
            for y in py:
                x_loc = np.int(np.round((x - px0) / shift))
                y_loc = np.int(np.round((y - py0) / shift))
                illu_obj = obj[y:y+probe_size, x:x+probe_size]
                g_osp = illu_obj * probe
                g_fsp = fftshift(fft2(g_osp))
                gp_fsp = obj_diff[y_loc,x_loc] * np.exp(1j*np.angle(g_fsp))
                gp_osp = ifft2(ifftshift(gp_fsp))
                obj_upd = illu_obj + (gp_osp - g_osp) * alpha * np.conj(probe) / np.max(np.abs(probe))**2
                obj[y:y+probe_size, x:x+probe_size] = obj_upd
                #probe =  probe + (gp_osp - g_osp) * kwargs['beta'] * np.conj(illu_obj) / np.max(np.abs(illu_obj))**2
    return obj, probe


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
    pxn, pyn = px0 + rows*shift, py0 + cols*shift
    px = np.arange(px0, pxn, shift)
    py = np.arange(py0, pyn, shift)


    probe = gau_kern(probe_size, probe_size / 2.35482,
            normalized=False)
    probe = probe > 0.5
    probe = probe.astype(np.int)

    obj_diff = diffract(probe, px, py, obj, **params)
    a, b = pie(probe, px, py, obj_diff, **params)

    fig,ax = plt.subplots(1,4)
    ax[0].imshow(np.abs(a), cmap='gray')
    ax[1].imshow(np.abs(obj), cmap='gray')
    ax[2].imshow(np.angle(a), cmap='gray')
    ax[3].imshow(np.angle(obj), cmap='gray')
    plt.show()

