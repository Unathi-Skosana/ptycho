
if __name__ == "__main__":
    from skimage.io import imread, imsave
    from skimage.transform import resize

    im = imread('images/USAF_Resolution_Chart_A2-780.jpg')
    im_resize = resize(im, (256, 256), anti_aliasing=True)
    imsave('images/USAF_Resolution_Chart_A2-256.tiff', im_resize)
