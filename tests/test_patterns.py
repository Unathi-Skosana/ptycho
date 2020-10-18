import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from numpy import pi

from utils import stripes, get_gradation_3d

if __name__ == "__main__":
    im1 = stripes(32, 16, 256, 256, horizontal=False)
    im2 = get_gradation_3d(256, 256, (0, 0, 0), (255, 255, 255), (False, True,
        False)).astype(np.uint8)

    fig, ax = plt.subplots(1,2)

    ax[0].imshow(im1)
    ax[1].imshow(im2)

    plt.show()
