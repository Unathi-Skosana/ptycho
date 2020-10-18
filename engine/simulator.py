"""
File: simulator.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
"""

import numpy as np

from joblib import Parallel, delayed
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from itertools import chain
from utils.filters import gau_kern


def diffract(obj, illu_func, illu_pos, mode='ideal', **kwargs):
    """
    Computes diffraction patterns of the object O ( phase & amplitude )
    probed by a probe P (phase & amplitude) across predetermined illuminations
    positions on the object.

    Args:
        obj: Object
        illu_func: Illumination function
        illu_pos: Illumination positions for the probe across the object O
        **kwargs: Arbitrary keyword arguments.

    Returns:
        List of diffraction patterns
    """

    # parameters
    pixels = kwargs['cols'] * kwargs['rows']
    rows = kwargs['rows']
    p_x, p_y = kwargs['p_x'], kwargs['p_y']
    shift = kwargs['shift']
    probe = kwargs['probe']

    # diffraction patterns
    diff_patterns = np.zeros((pixels, probe, probe), dtype=np.complex64)

    # Routine for parallelization, no need to worry about racing conditions as
    # the diffraction patterns are computed independently of one another
    def diff(k):
        dx_k, dy_k = 0, 0
        if mode =='position' and k % 6  == 0:
            dx_k = np.random.randint(low=-4, high=4)
            dy_k = np.random.randint(low=-4, high=4)

        x_k, y_k = illu_pos[k]
        i, j = np.int(np.round((x_k - p_x) / shift)), \
            np.int(np.round((y_k - p_y) / shift))

        ext_wave = obj[y_k + dy_k:y_k + dy_k + probe, x_k + dx_k:x_k + dx_k + probe] * illu_func
        ext_diff = np.abs(fftshift(fft2(ext_wave)))

        if mode == 'poisson':
            mu = 10e6
            vals = len(np.unique(ext_diff))
            vals = 2 ** np.ceil(np.log2(vals))
            ext_diff = np.random.poisson(vals  / mu * ext_diff) \
                    / float(vals / mu)
        if mode == 'random':
            v = 50
            def f(a):
                c = list(map(lambda I : I + np.random.uniform(low=-v/100 * I, high=v/100 * I), a))
                return np.array(c)

            b = np.fromiter(chain.from_iterable(f(i) for i in ext_diff),
                    dtype=ext_diff.dtype)

            ext_diff = np.reshape(b, ext_diff.shape)

        diff_patterns[i * rows + j] = ext_diff

    # Parallel diffraction pattern calculation
    Parallel(n_jobs=8, prefer="threads")(
        delayed(diff)(k) for k in range(pixels))

    return diff_patterns


def epie(obj, illu_pos, diff_patterns, **kwargs):
    """
    Args:
        obj: Object
        diff_patterns: Diffraction pattern
        illu_pos: Illumination positions for the probe across the object O
        **kwargs: Arbitrary keyword arguments.
    Returns:
        Estimated object, estimated probe, root mean square error and
        sum of least squares error of the estimate

    """

    # parameters
    p_x = kwargs['p_x']
    p_y = kwargs['p_y']
    iterations = kwargs['iterations']
    rows = kwargs['rows']
    cols = kwargs['cols']
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    shift = kwargs['shift']
    probe = kwargs['probe']
    hold = kwargs['hold']
    rms_n = kwargs['rms_n']
    permute = kwargs['permute']

    # loop temp variables
    rms_i = 0
    rms_save = iterations // rms_n
    half_rc = rows * cols // 2

    # diffraction
    idx = range(rows * cols)

    # object shape
    height, width = obj.shape

    # object estimation initial guess
    obj_est = np.zeros((height, width), dtype=np.complex64)

    # illumination function initial guess
    illu_func_est = gau_kern(probe, probe / 2.35482, normalize=False)
    illu_func_est = illu_func_est > 0.5
    illu_func_est = illu_func_est.astype(np.int)

    # initialization for SSE errors
    sse = np.zeros(iterations)

    # holder variable for the guessed exit diffraction pattern
    ext_diff_sse = None

    # holder variable for the estimated object after some iterations
    obj_est_n = np.zeros((rms_n, height, width), dtype=np.complex64)

    k = 0

    while k < iterations:
        if permute:
            idx = np.random.permutation(idx)

        for i in idx:
            x_i, y_i = illu_pos[i]
            x_loc, y_loc = np.int(np.round((x_i - p_x) / shift)), \
                np.int(np.round((y_i - p_y) / shift))

            # steps 1 - 7 from doi:10.1016/j.ultramic.2004.11.006
            obj_g = obj_est[y_i:y_i+probe, x_i:x_i+probe]
            obj_g_cpy = np.copy(obj_g)
            ext_wave_g = obj_g * illu_func_est
            ext_diff_g = fftshift(fft2(ext_wave_g))
            ext_diff_c = diff_patterns[x_loc * rows + y_loc] * \
                np.exp(1j * np.angle(ext_diff_g))
            ext_wave_c = ifft2(ifftshift(ext_diff_c))
            ext_wave_upd = obj_g + (ext_wave_c - ext_wave_g) * alpha * \
                np.conj(illu_func_est) \
                / np.power(np.max(np.abs(illu_func_est)), 2)
            obj_est[y_i:y_i+probe, x_i:x_i+probe] = ext_wave_upd

            if k >= hold:
                illu_func_est = illu_func_est + (ext_wave_c - ext_wave_g) \
                    * beta * np.conj(obj_g_cpy) \
                    / np.power(np.max(np.abs(obj_g_cpy)), 2)

            # arbitrary
            if x_loc * rows + y_loc == half_rc:
                ext_diff_sse = ext_diff_g

        err = np.abs(np.power(np.abs(diff_patterns[half_rc]), 2)
                     - np.power(np.abs(ext_diff_sse), 2))

        sse[k] = np.sum(np.sum(np.power(err, 2))) / (height * width)

        if k % rms_save == 0:
            obj_est_n[rms_i] = obj_est
            rms_i += 1

        k += 1

    def gamma(obj_est_n):
        g_fac = np.sum(obj * np.conj(obj_est_n)) \
                / np.sum(np.power(np.abs(obj_est_n), 2))
        return np.sum(np.power(np.abs(obj - g_fac * obj_est_n), 2)) \
            / np.sum(np.power(np.abs(obj), 2))

    rms = np.array(list(map(gamma, obj_est_n)))

    return obj_est, illu_func_est, rms, sse


def pie(obj, illu_func, illu_pos, diff_patterns, **kwargs):
    """
    Args:
        obj: Object
        illu_func: Illumination function
        illu_pos: Illumination positions for the probe across the object O
        diff_patterns: Diffraction pattern
        **kwargs: Arbitrary keyword arguments.
    Returns:
        Estimated object, root mean square error and sum of least squares error
        of the estimate
    """

    # parameters
    p_x = kwargs['p_x']
    p_y = kwargs['p_y']
    iterations = kwargs['iterations']
    rows = kwargs['rows']
    cols = kwargs['cols']
    alpha = kwargs['alpha']
    shift = kwargs['shift']
    probe = kwargs['probe']
    permute = kwargs['permute']
    rms_n = kwargs['rms_n']

    # loop temp variables
    rms_i = 0
    rms_save = iterations // rms_n
    half_rc = rows * cols // 2

    # diffraction
    idx = range(rows * cols)

    # object shape
    height, width = obj.shape

    # object estimation initial guess
    obj_est = np.zeros((height, width), dtype=np.complex64)

    # initialization for SSE errors
    sse = np.zeros(iterations)

    # holder variable for the guessed exit diffraction pattern
    ext_diff_sse = None

    # holder variable for the estimated object after some iterations
    obj_est_n = np.zeros((rms_n, height, width), dtype=np.complex64)

    k = 0
    while k < iterations:
        if permute:
            idx = np.random.permutation(idx)

        for i in idx:
            x_i, y_i = illu_pos[i]
            x_loc, y_loc = np.int(np.round((x_i - p_x) / shift)),  \
                np.int(np.round((y_i - p_y) / shift))

            # steps 1 - 7 from doi:10.1016/j.ultramic.2004.11.006
            obj_g = obj_est[y_i:y_i+probe, x_i:x_i+probe]
            ext_wave_g = obj_g * illu_func
            ext_diff_g = fftshift(fft2(ext_wave_g))
            ext_diff_c = diff_patterns[x_loc * rows + y_loc] * \
                np.exp(1j * np.angle(ext_diff_g))
            ext_wave_c = ifft2(ifftshift(ext_diff_c))
            ext_wave_upd = obj_g + (ext_wave_c - ext_wave_g) * alpha * \
                np.conj(illu_func) / np.power(np.max(np.abs(illu_func)), 2)
            obj_est[y_i:y_i+probe, x_i:x_i+probe] = ext_wave_upd

            # arbitrary
            if x_loc * rows + y_loc == half_rc:
                ext_diff_sse = ext_diff_g

        err = np.abs(np.power(np.abs(diff_patterns[half_rc]), 2)
                     - np.power(np.abs(ext_diff_sse), 2))

        sse[k] = np.sum(np.sum(np.power(err, 2))) / (height * width)

        if k % rms_save == 0:
            obj_est_n[rms_i] = obj_est
            rms_i += 1

        k += 1

    def gamma(obj_est_n):
        g_fac = np.sum(obj * np.conj(obj_est_n)) \
                / np.sum(np.power(np.abs(obj_est_n), 2))
        return np.sum(np.power(np.abs(obj - g_fac * obj_est_n), 2)) \
            / np.sum(np.power(np.abs(obj), 2))

    rms = np.array(list(map(gamma, obj_est_n)))

    return obj_est, rms, sse
