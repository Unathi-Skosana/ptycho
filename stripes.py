import os
import sys
import matplotlib
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize
from numpy import pi
from utils import stripes, get_gradation_3d

if __name__ == "__main__":
    #im = stripes(32, 16, 256, horizontal=False)

    im = get_gradation_3d(256, 256, (0, 0, 0), (255, 255, 255), (False, True,
        False)).astype(np.uint8)

    fig,ax = plt.subplots()
    plt.imsave("./images/gradient.tiff", im, cmap='gray')
