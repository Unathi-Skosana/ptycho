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

    # parameter
    wave_length =  0.5e-6
    k = 2 * pi / wave_length
    pixel_size = 0.5e-6
    NA = 0.1
    cutoff = NA * k

    im_ft = fftshift(fft2(im_amp))
    m,n = im_ft.shape
    kx = np.arange(-pi/pixel_size, pi/pixel_size, 2*pi / pixel_size / n)
    ky = np.arange(-pi/pixel_size, pi/pixel_size, 2*pi / pixel_size / m)
    kxv, kyv = np.meshgrid(kx, ky)

    freq = kxv**2 + kyv**2
    sft = np.copy(freq)
    ctf = freq <= cutoff**2
    ctf = ctf.astype(int)

    cpsf = fftshift(ifft2(ifftshift(ctf)))
    ipsf = np.abs(cpsf)**2
    otf  = np.abs(fftshift(fft2(ifftshift(ipsf))))

    out_ft = np.multiply(im_ft, ctf)
    out_amp = np.abs(ifft2(ifftshift(out_ft)))
    out_int = out_amp**2

    out_ft_otf = np.multiply(im_ft, otf)
    out_amp_otf = np.abs(ifft2(ifftshift(out_ft_otf)))

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2,3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                )

    for ax, img in zip(grid,
            [im_amp,out_amp,out_amp_otf,np.log(np.abs(im_ft)),np.log(np.abs(out_ft)),otf]):
        # Iterating over the grid returns the Axes.
        ax.imshow(img, cmap='gray')
    plt.show()
