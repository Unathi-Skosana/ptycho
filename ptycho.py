import numpy as np

from joblib import Parallel, delayed
from utils import gau_kern
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def diffract(obj, illu_func, illu_pos, **kwargs):
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
    m = kwargs['c'] * kwargs['r']
    r = kwargs['r']
    px, py = kwargs['px'], kwargs['py']
    shift = kwargs['shift']
    probe = kwargs['probe']

    # diffraction patterns
    diff_patterns = np.zeros((m, probe, probe), dtype=np.complex64)

    # Routine for parallelization, no need to worry about racing conditions as
    # the diffraction patterns are computed independently of one another
    def d(k):
        x, y = illu_pos[k]
        i, j = np.int(np.round((x - px) / shift)), \
            np.int(np.round((y - py) / shift))
        ext_wave = obj[j:j+probe, i:i+probe] * illu_func
        diff_patterns[i * r + j] = np.abs(fftshift(fft2(ext_wave)))

    # Parallel diffraction pattern calculation
    Parallel(n_jobs=4, prefer="threads")(delayed(d)(k) for k in range(m))

    return diff_patterns


def epie(obj, illu_pos, diff_patterns, hold=10, rms_n=10, permute=False, **kwargs):
    """
    Args:
        obj: Object
        diff_patterns: Diffraction pattern
        illu_pos: Illumination positions for the probe across the object O
        hold: Number of iterations to withhold updating the estimated
                illumination function
        rms_n: Size of the RMS errors array
        permute:  Permute the retrieval of diffraction patterns
        **kwargs: Arbitrary keyword arguments.

    Returns:
    """

    # parameters
    px = kwargs['px']
    py = kwargs['py']
    N = kwargs['N']
    r = kwargs['r']
    c = kwargs['c']
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    shift = kwargs['shift']
    probe = kwargs['probe']

    # loop temp variables
    o = 0
    k = 0
    m = N // rms_n
    half_rc = r * c // 2

    # diffraction
    idx = range(r * c)

    # object shape
    h, w = obj.shape

    # object estimation initial guess
    obj_est = np.zeros((h, w), dtype=np.complex64)

    # illumination function initial guess
    illu_func_est = gau_kern(probe, probe / 2.35482, normalized=False)
    illu_func_est = illu_func_est > 0.5
    illu_func_est = illu_func_est.astype(np.int)

    # initialization for SSE errors
    sse = np.zeros(N)

    # holder variable for the guessed exit diffraction pattern
    ext_diff_sse = None

    # holder variable for the estimated object after some iterations
    obj_est_n = np.zeros((rms_n, h, w), dtype=np.complex64)

    while k < N:
        if permute:
            idx = np.random.permutation(idx)

        for u in idx:
            x, y = illu_pos[u]
            i, j = np.int(np.round((x - px) / shift)), \
                np.int(np.round((y - py) / shift))

            # steps 1 - 7 from doi:10.1016/j.ultramic.2004.11.006
            obj_g = obj_est[y:y+probe, x:x+probe]
            obj_g_cpy = np.copy(obj_g)
            ext_wave_g = obj_g * illu_func_est
            ext_diff_g = fftshift(fft2(ext_wave_g))
            ext_diff_c = diff_patterns[i * r + j] * \
                np.exp(1j * np.angle(ext_diff_g))
            ext_wave_c = ifft2(ifftshift(ext_diff_c))
            ext_wave_upd = obj_g + (ext_wave_c - ext_wave_g) * alpha * \
                np.conj(illu_func_est) \
                / np.power(np.max(np.abs(illu_func_est)), 2)
            obj_est[y:y+probe, x:x+probe] = ext_wave_upd

            if k >= hold:
                illu_func_est = illu_func_est + (ext_wave_c - ext_wave_g) \
                    * beta * np.conj(obj_g_cpy) \
                    / np.power(np.max(np.abs(obj_g_cpy)), 2)

            if i * r + j == half_rc: # arbitrary
                ext_diff_sse = ext_diff_g

        e = np.abs(np.power(np.abs(diff_patterns[r * c // 2]), 2)
                   - np.power(np.abs(ext_diff_sse), 2))

        sse[k] = np.sum(np.sum(np.power(e, 2))) / (h*w)

        if k % m == 0:
            obj_est_n[o] = obj_est
            o += 1

        k += 1

    def gamma(obj_est_n):
        g = np.sum(obj * np.conj(obj_est_n)) \
                / np.sum(np.power(np.abs(obj_est_n), 2))
        return np.sum(np.power(np.abs(obj - g * obj_est_n), 2)) \
            / np.sum(np.power(np.abs(obj), 2))

    rms = np.array(list(map(gamma, obj_est_n)))

    return obj_est, illu_func_est, rms, sse


def pie(obj, illu_func, illu_pos, diff_patterns, rms_n=10, permute=False, **kwargs):
    """
    Args:
        obj: Object
        illu_func: Illumination function
        illu_pos: Illumination positions for the probe across the object O
        diff_patterns: Diffraction pattern
        rms_n: Size of the RMS errors array
        hold: Number of iterations to withhold updating the estimated 
                illumination function
        permute:  Permute the retrieval of diffraction patterns
        **kwargs: Arbitrary keyword arguments.

    Returns: 
    """

    # parameters
    px = kwargs['px']
    py = kwargs['py']
    N = kwargs['N']
    r = kwargs['r']
    c = kwargs['c']
    alpha = kwargs['alpha']
    shift = kwargs['shift']
    probe = kwargs['probe']

    # loop temp variables
    o = 0
    k = 0
    m = N // rms_n
    half_rc = r * c // 2

    # diffraction
    idx = range(r * c)

    # object shape
    h, w = obj.shape

    # object estimation initial guess
    obj_est = np.zeros((h, w), dtype=np.complex64)

    # initialization for SSE errors
    sse = np.zeros(N)

    # holder variable for the guessed exit diffraction pattern
    ext_diff_sse = None

    # holder variable for the estimated object after some iterations
    obj_est_n = np.zeros((rms_n, h, w), dtype=np.complex64)

    while k < N:
        if permute:
            idx = np.random.permutation(idx)

        for u in idx:
            x, y = illu_pos[u]
            i, j = np.int(np.round((x - px) / shift)),  \
                np.int(np.round((y - py) / shift))

            # steps 1 - 7 from doi:10.1016/j.ultramic.2004.11.006
            obj_g = obj_est[y:y+probe, x:x+probe]
            ext_wave_g = obj_g * illu_func
            ext_diff_g = fftshift(fft2(ext_wave_g))
            ext_diff_c = diff_patterns[i * r + j] * \
                np.exp(1j * np.angle(ext_diff_g))
            ext_wave_c = ifft2(ifftshift(ext_diff_c))
            ext_wave_upd = obj_g + (ext_wave_c - ext_wave_g) * alpha * \
                np.conj(illu_func) / np.power(np.max(np.abs(illu_func)), 2)
            obj_est[y:y+probe, x:x+probe] = ext_wave_upd

            # arbitrary
            if i * r + j == half_rc:
                ext_diff_sse = ext_diff_g

        e = np.abs(np.power(np.abs(diff_patterns[r * c // 2]), 2)
                   - np.power(np.abs(ext_diff_sse), 2))

        sse[k] = np.sum(np.sum(np.power(e, 2))) / (h*w)

        if k % m == 0:
            obj_est_n[o] = obj_est
            o += 1

        k += 1

    def gamma(obj_est_n):
        g = np.sum(obj * np.conj(obj_est_n)) \
                / np.sum(np.power(np.abs(obj_est_n), 2))
        return np.sum(np.power(np.abs(obj - g * obj_est_n), 2)) \
            / np.sum(np.power(np.abs(obj), 2))

    rms = np.array(list(map(gamma, obj_est_n)))

    return obj_est, rms, sse
