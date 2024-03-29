if __name__ == "__main__":
    import argparse
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    from mpl_toolkits.axes_grid1 import ImageGrid
    from skimage import img_as_float
    from skimage.io import imread
    from skimage.transform import resize
    from numpy import pi
    from scipy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift

    from utils.filters import gau_kern
    from utils.aberrations import c_lpf, \
        inc_lpf, zernike_modes, \
        c_abber, inc_abber

    plt.style.use('mint')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--image", 
                        default="images/cameraman.tif",
                        help="Sample image")
    ARGS = parser.parse_args()

    im = img_as_float(imread(ARGS.image, as_gray=True))
    im = resize(im, (256, 256), anti_aliasing=True)
    im_amp = np.sqrt(im)
    im_ft = fftshift(fft2(im_amp))
    h,w = im_ft.shape

    # parameter
    wave_length =  0.5e-6
    k = 2 * pi / wave_length
    pixel_size = 0.5e-6
    NA = 0.1
    cutoff_freq = NA * k

    phi = zernike_modes([(-2,6,4)], 20, w)
    incoher_ft, incoher_otf = inc_abber(im_ft, \
            phi,
            pixel_size, \
            cutoff_freq)
    coher_aber_ft, coher_aber_ctf = c_abber(im_ft, \
            phi,
            pixel_size, \
            cutoff_freq)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(1,3),
                 axes_pad=0.1,
                )

    for ax, img in zip(grid,
                [
                im_amp,
                np.abs(ifft2(ifftshift(coher_aber_ft))),
                np.abs(ifft2(ifftshift(incoher_ft))),
                ]):
        ax.imshow(img, cmap='gray')

    fig1, ax1 = plt.subplots()
    ax1.imshow(phi, cmap='gray')

    plt.show()
