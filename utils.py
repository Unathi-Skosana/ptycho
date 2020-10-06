import numpy as np
from numpy import pi

# --- formatting --- #
flatten = lambda l: np.array([item for sl in l for item in sl])
vectorize_im = lambda im,m: flatten(im.reshape((m,1)))

def bmatrix(a):
    """Returns a LaTeX bmatrix
    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

# --- abberations --- #
from zernike import RZern
from operator import itemgetter
from skimage.transform import resize


def stripes(gap,stripe_width,image_size):
    canvas = np.full((image_size, image_size), 0.5, dtype=np.float64)
    current_col = 0
    while current_col < image_size:
        if current_col + stripe_width + gap <= image_size-1:
            canvas[:, current_col:current_col+stripe_width] = 1.0
            current_col += stripe_width + gap
        elif current_col + stripe_width <= image_size-1:
            canvas[:, current_col:current_col+stripe_width] = 1.0
            current_col = image_size
        else:
            canvas[:, current_col:] = 1
            current_col = image_size
    return canvas

def checkerboard(N,n):
    """
        N: size of board; n=size of each square; N/(2*n) must be an integer
        from https://stackoverflow.com/questions/32704485/drawing-a-checkerboard-in-python
    """
    if (N%(2*n)):
        print('Error: N/(2*n) must be an integer')
        return False
    a = np.concatenate((np.zeros(n),np.ones(n)))
    b = np.pad(a,int((N**2)/2-n),'wrap').reshape((N,N))
    return (b+b.T==1).astype(int)

def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def zern_modes(coefficients, radius, width):
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    nn = max(coefficients, key=itemgetter(1))[1]
    cart = RZern(nn)
    ddx = np.mgrid[-1:1:width*1j]
    ddy = np.mgrid[-1:1:width*1j]
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    c = np.zeros(cart.nk)

    for coeff in coefficients:
        m,n,v = coeff
        j =  int(n * (n + 1) / 2 + np.abs(m))
        if m >= 0 and n % 4 == 2 or n % 4 == 3:
            j += 1
        if m <= 0 and n % 4 == 0 or n % 4 == 1:
            j += 1
        c[j-1] = v
    phi = np.nan_to_num(cart.eval_grid(c, matrix=True), nan=0)

    if width != radius:
        phi = np.pad(phi,width//2 - radius//2,pad_with,padder=0)

    return resize(phi, (width, width), anti_aliasing=False)


# --- filters --- #
from scipy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift

def c_lpf(im_ft,pixel_size,cutoff_freq):
    m,n = im_ft.shape
    kx = np.arange(-pi/pixel_size, pi/pixel_size, 2*pi / pixel_size / n)
    ky = np.arange(-pi/pixel_size, pi/pixel_size, 2*pi / pixel_size / m)
    kxv, kyv = np.meshgrid(kx, ky)

    freq = kxv**2 + kyv**2
    ctf = freq <= cutoff_freq**2
    out_ft = np.multiply(im_ft, ctf)

    return out_ft,ctf.astype(np.uint)

def inc_lpf(im_ft,pixel_size,cutoff_freq):
    _,ctf = c_lpf(im_ft, pixel_size, cutoff_freq)
    cpsf = fftshift(ifft2(ifftshift(ctf)))
    ipsf = np.abs(cpsf)**2
    otf  = np.abs(fftshift(fft2(ifftshift(ipsf))))
    otf = otf/np.max(otf)
    out_ft = np.multiply(im_ft, otf)

    return out_ft,otf

def c_abber(phi,im_ft,pixel_size,cutoff_freq):
    _, ctf = c_lpf(im_ft,pixel_size,cutoff_freq)
    aber_ctf = np.multiply(np.exp(1j * phi), ctf)
    aber_ft = np.multiply(im_ft, aber_ctf)

    return aber_ft, aber_ctf

def inc_abber(phi,im_ft,pixel_size,cutoff_freq):
    _,aber_ctf = c_abber(phi,im_ft,pixel_size,cutoff_freq)
    cpsf = fftshift(ifft2(ifftshift(aber_ctf)))
    ipsf = np.abs(cpsf)**2
    otf  = np.abs(fftshift(fft2(ifftshift(ipsf))))
    otf = otf/np.max(otf)
    out_ft = np.multiply(im_ft, otf)

    return out_ft,otf

# --- misc --- #
from scipy import signal
# https://gist.github.com/thomasaarholt/267ec4fff40ca9dff1106490ea3b75670

def gau_kern(n, std, normalized=False):
    '''
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalized,
    its volume equals 1.
    '''

    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalized:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D


def circ_mask(c, r, w, v=0):
    '''
    Generate circular mask centered at c with radius r
    '''

    cx, cy = c
    y,x = np.ogrid[-cx:w-cx, -cy:w-cy]
    mask = x*x + y*y <= r*r
    arr = np.zeros((w,w))
    arr[mask] = v
    return arr
