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
from utils import stripes

# TeX typesetting
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Aesthetics
plt.close('all')
plt.style.use('seaborn-pastel')

if __name__ == "__main__":
    im = stripes(32, 32, 256)
    fig,ax = plt.subplots()
    plt.imsave("./images/r_stripes.tiff", im, cmap='gray_r')
