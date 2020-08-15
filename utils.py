import numpy as np
from numpy import pi
from scipy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift

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

def wavefront_abber(coefficients, width, height):
    nn = max(coefficients, key=itemgetter(1))[1]
    cart = RZern(nn)
    ddx = np.linspace(-1.0, 1.0, width)
    ddy = np.linspace(-1.0, 1.0, height)
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
    phi = cart.eval_grid(c, matrix=True)
    return np.nan_to_num(phi, nan=1.0)


# --- filters --- #
def coherent_low_pass_filter(im_ft,pixel_size,cutoff_freq):
    m,n = im_ft.shape
    kx = np.arange(-pi/pixel_size, pi/pixel_size, 2*pi / pixel_size / n)
    ky = np.arange(-pi/pixel_size, pi/pixel_size, 2*pi / pixel_size / m)
    kxv, kyv = np.meshgrid(kx, ky)

    freq = kxv**2 + kyv**2
    ctf = freq <= cutoff_freq**2
    out_ft = np.multiply(im_ft, ctf)

    return out_ft,ctf

def incoherent_low_pass_filter(im_ft,pixel_size,cutoff_freq):
    _,ctf = coherent_low_pass_filter(im_ft, pixel_size, cutoff_freq)
    cpsf = fftshift(ifft2(ifftshift(ctf)))
    ipsf = np.abs(cpsf)**2
    otf  = np.abs(fftshift(fft2(ifftshift(ipsf))))
    otf = otf/np.max(otf)
    out_ft = np.multiply(im_ft, otf)

    return out_ft,otf


def coherent_low_pass_filter_abber(phi,im_ft,pixel_size,cutoff_freq):
    _, ctf = coherent_low_pass_filter(im_ft,pixel_size,cutoff_freq)
    aber_ctf = np.multiply(np.exp(1j * phi), ctf)
    aber_ft = np.multiply(im_ft, aber_ctf)

    return aber_ft, aber_ctf

def incoherent_low_pass_filter_abber(phi, im_ft,pixel_size,cutoff_freq):
    _,aber_ctf = coherent_low_pass_filter_abber(phi, im_ft, pixel_size, cutoff_freq)
    cpsf = fftshift(ifft2(ifftshift(aber_ctf)))
    ipsf = np.abs(cpsf)**2
    otf  = np.abs(fftshift(fft2(ifftshift(ipsf))))
    out_ft = np.multiply(im_ft, otf)

    return out_ft,otf
