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
from utils import circ_mask

# TeX typesetting
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Aesthetics
plt.close('all')
plt.style.use('seaborn-pastel')

if __name__ == "__main__":
    im_path = sys.argv[1]
    im = img_as_float(imread(im_path, as_gray=True))
    im = resize(im, (256, 256), anti_aliasing=True)

    mask = circ_mask((256//4+50, 256//4-50), 30, 256, v=1.0)
    fig,ax = plt.subplots()
    ax.imshow(np.multiply(im, mask), cmap='gray')
    plt.show()
