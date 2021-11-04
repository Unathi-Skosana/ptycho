import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from skimage.io import imread
from skimage import img_as_float

plt.style.use("mint")

im1 = img_as_float(imread("./v10_obj_ampl.png", as_gray=True))
im2 = img_as_float(imread("./v10_obj_phase.png", as_gray=True))
im3 = img_as_float(imread("./v10_probe_ampl.png", as_gray=True))
im4 = img_as_float(imread("./v10_probe_phase.png", as_gray=True))
im5 = img_as_float(imread("./v30_obj_ampl.png", as_gray=True))
im6 = img_as_float(imread("./v30_obj_phase.png", as_gray=True))
im7 = img_as_float(imread("./v30_probe_ampl.png", as_gray=True))
im8 = img_as_float(imread("./v30_probe_phase.png", as_gray=True))
im9 = img_as_float(imread("./v50_obj_ampl.png", as_gray=True))
im10 = img_as_float(imread("./v50_obj_phase.png", as_gray=True))
im11 = img_as_float(imread("./v50_probe_ampl.png", as_gray=True))
im12 = img_as_float(imread("./v50_probe_phase.png", as_gray=True))

fig = plt.figure(figsize=(16, 16))
grid = ImageGrid(
    fig,
    111,  # similar to subplot(111)
    nrows_ncols=(2, 6),  # creates 2x2 grid of axes
    axes_pad=0.1,  # pad between axes in inch.
)

i = 0
j = 0
for ax, im in zip(
    grid, [im1, im2, im5, im6, im9, im10, im3, im4, im7, im8, im11, im12]
):
    if i % 2 == 0 and i < 6:
        if j == 0:
            ax.set_title(r"$\nu = 0.1$", size=40, x=1.0)
        if j == 1:
            ax.set_title(r"$\nu = 0.3$", size=40, x=1.0)
        if j == 2:
            ax.set_title(r"$\nu = 0.5$", size=40, x=1.0)
        j += 1

    ax.set_axis_off()
    ax.imshow(im, cmap="gray")
    i += 1

fig.savefig('random_grid.png', bbox_inches='tight',
                 pad_inches=0, transparent=False)
plt.show()
