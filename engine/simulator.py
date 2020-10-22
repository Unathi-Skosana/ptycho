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
from itertools import chain, product
from utils.filters import gau_kern


class PytchoSimulatorBase():
    """
    docs
    """

    def __init__(self, alpha, beta, probe,
                 start, shift, rc, iterations):

        self._alpha = alpha
        self._beta = beta
        self._probe = probe
        self._start = start
        self._shift = shift
        self._rc = rc
        self._iterations = iterations

        self.compute_illu_pos()

    @property
    def alpha(self):
        """
        docs
        """

        return self._alpha

    @alpha.setter
    def alpha(self, _alpha):
        """
        docs
        """

        self._alpha = _alpha

    @property
    def beta(self):
        """
        docs
        """

        return self._beta

    @beta.setter
    def beta(self, _beta):
        """
        docs
        """
        self._beta = _beta

    @property
    def probe(self):
        """
        docs
        """
        return self._probe

    @probe.setter
    def probe(self, _probe):
        self._probe = _probe

    @property
    def shift(self):
        """
        docs
        """
        return self._shift

    @shift.setter
    def shift(self, _shift):
        """
        docs
        """

        self._shift = _shift

    @property
    def start(self):
        """
        docs
        """
        return self._start

    @start.setter
    def start(self, _start):
        self._start = _start

    @property
    def rc(self):
        """
        docs
        """
        return self._rc

    @rc.setter
    def rc(self, _rc):
        self._rc = _rc


    @property
    def iterations(self):
        """
        docs
        """
        return self._iterations

    @iterations.setter
    def iterations(self, _iterations):
        self._iterations = _iterations


    def compute_illu_pos(self):
        s_x, s_y = self._start
        rows, cols = self._rc
        x = np.arange(s_x, s_x + rows * self._shift, self._shift)
        y = np.arange(s_y, s_y + cols * self._shift, self._shift)

        self._illu_pos = np.array(list(product(x, y)))


class PytchoSimulator(PytchoSimulatorBase):
    def __init__(self, alpha=1., beta=1., probe=50,
                 start=(2, 2), shift=20, rc=(11, 11),
                 iterations=200):
        super().__init__(alpha=alpha, beta=beta,
                         probe=probe, start=start,
                         shift=shift, rc=rc,
                         iterations=iterations)

    def diffract(self, obj, illu_func, mode='', **kwargs):
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

        '''
        allowedtypes = {
            'poisson': 'poisson_values',
            'random': 'random_values'
        }

        kwdefaults = {
            'mean': 0.,
            'amount': 00
        }

        allowedkwargs = {
            'poisson': ['mean'],
            'random': ['amount'],
        }

        for key in kwargs:
            if key not in allowedkwargs[allowedtypes[mode]]:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                (key, allowedkwargs[allowedtypes[mode]]))
        '''

        # diffraction patterns
        diff_patterns = np.zeros((np.prod(self.rc), self._probe, self._probe),
                                 dtype=np.complex64)

        # Routine for parallelization, no need to worry about racing conditions
        # as the diffraction patterns are computed independently of one another
        def diff(k):
            dx_k, dy_k = 0, 0

            if mode == 'position' and k % 5 == 0:
                dx_k = np.random.randint(low=-2, high=2)
                dy_k = np.random.randint(low=-2, high=2)

            x_k, y_k = self._illu_pos[k]
            i, j = np.int(np.round((x_k - self._start[0]) / self._shift)), \
                np.int(np.round((y_k - self._start[1]) / self._shift))

            ext_wave = obj[y_k + dy_k:y_k + dy_k + self._probe,
                           x_k + dx_k:x_k + dx_k + self._probe] * illu_func
            ext_diff = np.abs(fftshift(fft2(ext_wave)))


            if mode == 'poisson':
                mu = kwargs['mean']
                ext_inten = ext_diff ** 2
                fac = mu / np.sum(ext_inten)
                ext_inten_noisy = np.random.poisson(ext_inten * fac) / fac
                ext_diff = np.sqrt(ext_inten_noisy)

            if mode == 'random':
                v = kwargs['amount']
                if not 0 <= v <= 1.0:
                    raise ValueError('Mean must be between 0 and 1.0 for random \
                            noise.')

                def f(col):
                    noisy_col = list(map(lambda I: I +
                                         np.random.uniform(low=-v * I, high=v * I),
                                         col))
                    return np.array(noisy_col)

                noisy_vec = np.fromiter(chain.from_iterable(f(i) for i in ext_diff),
                                        dtype=ext_diff.dtype)

                ext_diff = np.reshape(noisy_vec, ext_diff.shape)

            diff_patterns[i * self.rc[0] + j] = ext_diff

        # Parallel diffraction pattern calculation
        Parallel(n_jobs=8, prefer="threads")(
            delayed(diff)(k) for k in range(np.prod(self._rc)))

        return diff_patterns

    def epie(self, obj, diff_patterns, **kwargs):
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
        hold = kwargs.get('hold', 10)
        err_ival = kwargs.get('err_ival', 4)
        permute = kwargs.get('permute', False)

        # loop temp variables
        err_i = 0
        err_n = int(np.ceil(self._iterations / err_ival))
        half_rc = np.prod(self._rc) // 2

        # diffraction
        idx = range(np.prod(self._rc))

        # object shape
        height, width = obj.shape

        # object estimation initial guess
        obj_est = np.zeros((height, width), dtype=np.complex64)

        # illumination function initial guess
        illu_func_est = np.ones((self._probe, self._probe),
                dtype=np.complex64) * np.pi

        # initialization for SSE errors
        sse = np.zeros(err_n)

        # holder variable for the guessed exit diffraction pattern
        ext_diff_sse = None

        # holder variable for the estimated object after some iterations
        obj_est_n = np.zeros((err_n, height, width), dtype=np.complex64)

        k = 0

        while k < self._iterations:
            if permute:
                idx = np.random.permutation(idx)

            for i in idx:
                x_i, y_i = self._illu_pos[i]
                x_loc = np.int(np.round((x_i - self._start[0]) / self._shift))
                y_loc = np.int(np.round((y_i - self._start[1]) / self._shift))

                # steps 1 - 7 from doi:10.1016/j.ultramic.2004.11.006
                obj_g = obj_est[y_i:y_i+self._probe, x_i:x_i+self._probe]
                obj_g_cpy = np.copy(obj_g)
                ext_wave_g = obj_g * illu_func_est
                ext_diff_g = fftshift(fft2(ext_wave_g))
                ext_diff_c = diff_patterns[x_loc * self.rc[0] + y_loc] \
                    * np.exp(1j * np.angle(ext_diff_g))
                ext_wave_c = ifft2(ifftshift(ext_diff_c))

                ext_wave_upd = obj_g + (ext_wave_c - ext_wave_g) \
                    * 0.2 * np.conj(illu_func_est) \
                    / ((1. - self._alpha) * np.power(np.abs(illu_func_est), 2) \
                    + self._alpha * np.power(np.max(np.abs(illu_func_est)), 2))
                obj_est[y_i:y_i+self._probe, x_i:x_i+self._probe] = ext_wave_upd

                if k >= hold:
                    illu_func_est = illu_func_est + (ext_wave_c - ext_wave_g) \
                        * 1.0 * np.conj(obj_g_cpy) \
                        / ((1 - self._beta) * np.power(np.abs(obj_g_cpy), 2) \
                        + self._beta * np.power(np.max(np.abs(obj_g_cpy)), 2))

                # arbitrary
                if x_loc * self.rc[0] + y_loc == half_rc:
                    ext_diff_sse = ext_diff_g

            if k % err_ival == 0:
                err = np.abs(np.power(np.abs(diff_patterns[half_rc]), 2)
                             - np.power(np.abs(ext_diff_sse), 2))
                sse[err_i] = np.sum(np.sum(np.power(err, 2))) / (height * width)
                obj_est_n[err_i] = obj_est
                err_i += 1

            k += 1

        def gamma(obj_est_n):
            g_fac = np.sum(obj * np.conj(obj_est_n)) \
                    / np.sum(np.power(np.abs(obj_est_n), 2))
            return np.sum(np.power(np.abs(obj - g_fac * obj_est_n), 2)) \
                / np.sum(np.power(np.abs(obj), 2))

        rms = np.array(list(map(gamma, obj_est_n)))

        return obj_est, illu_func_est, rms, sse

    def pie(self, obj, illu_func, diff_patterns, **kwargs):
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
        err_ival = kwargs.get('err_ival', 4)
        permute = kwargs.get('permute', False)


        # loop temp variables
        err_i = 0
        err_n = int(np.ceil(self._iterations / err_ival))
        half_rc = np.prod(self._rc) // 2

        # diffraction
        idx = range(np.prod(self.rc))

        # object shape
        height, width = obj.shape

        # object estimation initial guess
        obj_est = np.zeros((height, width), dtype=np.complex64)

        # initialization for SSE errors
        sse = np.zeros(err_n)

        # holder variable for the guessed exit diffraction pattern
        ext_diff_sse = None

        # holder variable for the estimated object after some iterations
        obj_est_n = np.zeros((err_n, height, width), dtype=np.complex64)

        k = 0
        while k < self._iterations:
            if permute:
                idx = np.random.permutation(idx)

            for i in idx:
                x_i, y_i = self._illu_pos[i]
                x_loc = np.int(np.round((x_i - self._start[0]) / self._shift))
                y_loc = np.int(np.round((y_i - self._start[1]) / self._shift))

                # steps 1 - 7 from doi:10.1016/j.ultramic.2004.11.006
                obj_g = obj_est[y_i:y_i+self._probe, x_i:x_i+self._probe]
                ext_wave_g = obj_g * illu_func
                ext_diff_g = fftshift(fft2(ext_wave_g))
                ext_diff_c = diff_patterns[x_loc * self._rc[0] + y_loc] * \
                    np.exp(1j * np.angle(ext_diff_g))
                ext_wave_c = ifft2(ifftshift(ext_diff_c))
                ext_wave_upd = obj_g + (ext_wave_c - ext_wave_g) \
                    * 0.5 * np.conj(illu_func) \
                    / ((1 - self._alpha)*np.power(np.max(np.abs(illu_func)),2) \
                    + self._alpha * np.power(np.max(np.abs(illu_func)), 2))
                obj_est[y_i:y_i+self._probe, x_i:x_i+self._probe] = ext_wave_upd

                # arbitrary
                if x_loc * self._rc[0] + y_loc == half_rc:
                    ext_diff_sse = ext_diff_g

            if k % err_ival == 0:
                err = np.abs(np.power(np.abs(diff_patterns[half_rc]), 2)
                             - np.power(np.abs(ext_diff_sse), 2))
                sse[err_i] = np.sum(np.sum(np.power(err, 2))) / (height * width)
                obj_est_n[err_i] = obj_est
                err_i += 1

            k += 1

        def gamma(obj_est_n):
            g_fac = np.sum(obj * np.conj(obj_est_n)) \
                    / np.sum(np.power(np.abs(obj_est_n), 2))
            return np.sum(np.power(np.abs(obj - g_fac * obj_est_n), 2)) \
                / np.sum(np.power(np.abs(obj), 2))

        rms = np.array(list(map(gamma, obj_est_n)))

        return obj_est, rms, sse
