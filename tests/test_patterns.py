import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from numpy import pi
from utils import stripes, get_gradation_2d,\
        get_gradation_3d, checkerboard, radial_gradient
from utils.filters import gau_kern, circ_mask
from skimage.color import rgb2gray
from skimage import img_as_float

if __name__ == "__main__":
    cmask = circ_mask(70, (35, 35), 35, 0)
    im1 = stripes(32, 45, 256, 256, horizontal=False)
    im2 = get_gradation_2d(256, 256, 0, 255, is_horizontal=False)
    im3 = checkerboard(256, 64)
    im4 = stripes(32, 45, 256, 256, horizontal=True)
    im5 = rgb2gray(radial_gradient(70, 70))

    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1, 5),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, [im1, im2, im3, im4, im5]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='gray')
        ax.set_axis_off()

    plt.imsave('vstripes.tiff', im1)
    plt.imsave('gradient_2d.tiff', im2)
    plt.imsave('checkerboard.tiff', im3)
    plt.imsave('hstripes.tiff', im4)
    plt.imsave('gradient_3d.tiff', im5)

    plt.show()
