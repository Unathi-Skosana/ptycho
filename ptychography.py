import os
import sys
import matplotlib
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from skimage import io,transform,color,img_as_float
from numpy import pi
from scipy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift
from scipy.signal import convolve
from utils import coherent_low_pass_filter, \
    incoherent_low_pass_filter, wavefront_abber, \
    coherent_low_pass_filter_abber, incoherent_low_pass_filter_abber

# TeX typesetting
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Aesthetics
plt.close('all')
plt.style.use('seaborn-pastel')

if __name__ == "__main__":
    im_path = sys.argv[1]
    im = img_as_float(io.imread(im_path, as_gray=True))
    im_amp =  np.sqrt(im)
    im_ft = fftshift(fft2(im_amp))
    h,w = im_ft.shape

    # parameter
    wave_length =  0.5e-6
    k = 2 * pi / wave_length
    pixel_size = 0.5e-6
    NA = 0.1
    cutoff_freq = NA * k

    phi = wavefront_abber([(0,4,2)], 20, w)
    incoher_ft, incoher_otf = incoherent_low_pass_filter_abber(phi, im_ft, \
            pixel_size, \
            cutoff_freq)
    coher_aber_ft, coher_aber_ctf = coherent_low_pass_filter_abber(phi, im_ft, \
            pixel_size, \
            cutoff_freq)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(3,2),
                 axes_pad=0.1,
                )

    for ax, img in zip(grid,
                [
                im_amp,
                np.abs(ifft2(ifftshift(coher_aber_ft))),
                np.abs(ifft2(ifftshift(incoher_ft))),
                np.log(np.abs(coher_aber_ctf)),
                np.log(np.abs(incoher_otf)),
                phi
                ]):
        # Iterating over the grid returns the Axes.
        ax.imshow(img, cmap='gray')
    plt.show()
