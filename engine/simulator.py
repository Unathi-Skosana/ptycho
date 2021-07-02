"""
File: simulator.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
"""

import numpy as np

from skimage.restoration import unwrap_phase
from skimage.exposure import rescale_intensity
from joblib import Parallel, delayed
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from itertools import chain, product
from utils.filters import gau_kern, circ_mask
from utils.patterns import radial_gradient
from skimage.color import rgb2gray
from skimage.util import invert

class PytchoSimulatorBase():
    """
    The base class of a ptychography simulator, holding the various
    parameters of the simulation.
    """

    def __init__(self, alpha, beta, probe,
                 start, shift, rc, iterations):
        """ Initialize class instance """
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
        """ Get parameter alpha """

        return self._alpha

    @alpha.setter
    def alpha(self, _alpha):
        """ Set the value of alpha """

        self._alpha = _alpha

    @property
    def beta(self):
        """ Get parameter beta """

        return self._beta

    @beta.setter
    def beta(self, _beta):
        """ Set the value of beta """

        self._beta = _beta

    @property
    def probe(self):
        """ Get probe size """

        return self._probe

    @probe.setter
    def probe(self, _probe):
        """ Set value of probe """

        self._probe = _probe

    @property
    def shift(self):
        """ Get probe shift """

        return self._shift

    @shift.setter
    def shift(self, _shift):
        """ Set value of probe shift """

        self._shift = _shift

    @property
    def start(self):
        """ Get start position of probe """

        return self._start

    @start.setter
    def start(self, _start):
        """ Set the value of start position """

        self._start = _start

    @property
    def rc(self):
        """ Get rows and columns of probe positions """

        return self._rc

    @rc.setter
    def rc(self, _rc):
        """ Set the value of rows and columns of probe positions """

        self._rc = _rc

    @property
    def iterations(self):
        """ Get number of iterations """

        return self._iterations

    @iterations.setter
    def iterations(self, _iterations):
        """ Set the value of iterations """

        self._iterations = _iterations

    def compute_illu_pos(self):
        """ Compute the illumination positions of the probe on the image"""

        s_x, s_y = self._start
        rows, cols = self._rc
        x = np.arange(s_x, s_x + rows * self._shift, self._shift)
        y = np.arange(s_y, s_y + cols * self._shift, self._shift)

        self._illu_pos = np.array(list(product(x, y)))


class PytchoSimulator(PytchoSimulatorBase):
    """
    A class for simulating a ptychographic image reconstructions
    """

    def __init__(self, alpha=1., beta=1., probe=50,
                 start=(2, 2), shift=20, rc=(11, 11),
                 iterations=200):
        """ Initialize class instance """
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
            dx, dy = 0, 0
            if mode == 'position':
                positions = kwargs.get('positions', range(5, 100))
                if k in positions:
                    dx = np.random.randint(low=-4, high=4)
                    dy = np.random.randint(low=-4, high=4)

            x_k, y_k = self._illu_pos[k]
            i = np.int(np.round((x_k - self._start[0]) / self._shift))
            j = np.int(np.round((y_k - self._start[1]) / self._shift))
            ext_wave = obj[y_k + dy:y_k + dy + self._probe,
                           x_k + dx:x_k + dx + self._probe] * illu_func
            ext_diff = np.abs(fftshift(fft2(ext_wave)))

            if mode == 'poisson':
                mu = kwargs.get('mean', 1e5)
                ext_inten = ext_diff ** 2
                fac = mu / np.sum(ext_inten)
                ext_inten_noisy = np.random.poisson(ext_inten * fac) / fac
                ext_diff = np.sqrt(ext_inten_noisy)

            if mode == 'random':
                mu = kwargs.get('mean', 1e6)
                ext_inten = ext_diff ** 2
                fac = mu / np.sum(ext_inten)
                ext_inten_noisy = np.random.poisson(ext_inten * fac) / fac
                ext_diff = np.sqrt(ext_inten_noisy)

                v = kwargs.get('amount', 0.05)
                if not 0 <= v <= 1.0:
                    raise ValueError('Mean must be between 0 and 1.0 for random \
                            noise.')

                # bottleneck
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
        hold = kwargs.get('hold', 1)
        err_ival = kwargs.get('err_ival', 1)
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
        illu_func_est = np.ones((self._probe, self._probe))

        # initialization for error
        R_factor = np.zeros(err_n)

        # holder variable for the estimated object after some iterations
        obj_est_n = np.zeros((err_n, height, width), dtype=np.complex64)

        k = 0

        diff_pat_sum = np.sum(np.abs(diff_patterns[half_rc])**2)
        MN = np.prod(diff_patterns[0].shape)

        while k < self._iterations:
            ext_waves = []
            ext_wave_diffs = []

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

                if k >= hold:
                    # probe power correction
                    illu_func_est = illu_func_est * np.sqrt(diff_pat_sum / (MN * np.sum(np.abs(illu_func_est)**2)))

                ext_wave_upd = obj_g + (ext_wave_c - ext_wave_g) \
                    * self._alpha * illu_func_est.conj() \
                    / np.max(np.abs(illu_func_est))**2
                obj_est[y_i:y_i+self._probe, x_i:x_i+self._probe] = ext_wave_upd

                ext_wave_diffs.append(diff_patterns[x_loc * self.rc[0] + y_loc])
                ext_waves.append(ext_diff_g)

                if k >= hold:
                    illu_func_est = illu_func_est + (ext_wave_c - ext_wave_g) \
                        * self._beta * obj_g_cpy.conj() \
                        / np.max(np.abs(obj_g_cpy))**2

            if k % err_ival == 0:
                ext_waves = np.array(ext_waves)
                ext_wave_diffs = np.array(ext_wave_diffs)
                numerator = np.abs(ext_wave_diffs - np.abs(ext_waves))
                denominator = np.abs(ext_wave_diffs)
                R_factor[err_i] = np.sum(np.sum(numerator)) / np.sum(np.sum(denominator))
                obj_est_n[err_i] = obj_est
                err_i += 1

            k += 1

        def gamma(obj_est_n):
            g_fac = np.sum(obj * obj_est_n.conj()) \
                    / np.sum(np.abs(obj_est_n)**2)
            return np.sum(np.abs(obj - g_fac * obj_est_n)**2) \
                / np.sum(np.abs(obj)**2)

        RMS = np.array(list(map(gamma, obj_est_n)))

        return obj_est, illu_func_est, RMS, R_factor


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

        # initialization R_factor
        R_factor = np.zeros(err_n)

        # holder variable for the estimated object after some iterations
        obj_est_n = np.zeros((err_n, height, width), dtype=np.complex64)

        gau = gau_kern(self._probe, self.probe / np.sqrt(8 * np.log(2)),
                normalize=False)

        k = 0

        while k < self._iterations:
            ext_waves = []
            ext_wave_diffs = []

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
                    * np.abs(illu_func) * illu_func.conj() \
                    / (np.max(np.abs(illu_func)) * \
                    (np.abs(illu_func)**2 + \
                    self._alpha * np.max(np.abs(illu_func))**2))
                obj_est[y_i:y_i+self._probe, x_i:x_i+self._probe] = ext_wave_upd

                ext_wave_diffs.append(diff_patterns[x_loc * self.rc[0] + y_loc])
                ext_waves.append(ext_diff_g)

            if k % err_ival == 0:
                ext_waves = np.array(ext_waves)
                ext_wave_diffs = np.array(ext_wave_diffs)
                numerator = np.abs(ext_wave_diffs - np.abs(ext_waves))
                denominator = np.abs(ext_wave_diffs)
                R_factor[err_i] = np.sum(np.sum(numerator)) / np.sum(np.sum(denominator))
                obj_est_n[err_i] = obj_est
                err_i += 1

            k += 1

        def gamma(obj_est_n):
            g_fac = np.sum(obj * obj_est_n.conj()) \
                    / np.sum(np.abs(obj_est_n)**2)
            return np.sum(np.abs(obj - g_fac * obj_est_n)**2) \
                / np.sum(np.abs(obj)**2)

        RMS = np.array(list(map(gamma, obj_est_n)))

        return obj_est, RMS, R_factor


    def rpie(self, obj, illu_func, diff_patterns, **kwargs):
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

        # initialization for R_factor
        R_factor = np.zeros(err_n)

        # holder variable for the estimated object after some iterations
        obj_est_n = np.zeros((err_n, height, width), dtype=np.complex64)

        gau = gau_kern(self._probe, self.probe / np.sqrt(8 * np.log(2)),
                normalize=False)
        k = 0

        while k < self._iterations:
            ext_waves = []
            ext_wave_diffs = []

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
                    * illu_func.conj() \
                    / ((1 - self._alpha) * np.max(np.abs(illu_func))**2 \
                    + self._alpha * np.max(np.abs(illu_func))**2)
                obj_est[y_i:y_i+self._probe, x_i:x_i+self._probe] = ext_wave_upd

                ext_wave_diffs.append(diff_patterns[x_loc * self.rc[0] + y_loc])
                ext_waves.append(ext_diff_g)

            if k % err_ival == 0:
                ext_waves = np.array(ext_waves)
                ext_wave_diffs = np.array(ext_wave_diffs)
                numerator = np.abs(ext_wave_diffs - np.abs(ext_waves))
                denominator = np.abs(ext_wave_diffs)
                R_factor[err_i] = np.sum(np.sum(numerator)) / np.sum(np.sum(denominator))
                obj_est_n[err_i] = obj_est
                err_i += 1

            k += 1

        def gamma(obj_est_n):
            g_fac = np.sum(obj * obj_est_n.conj()) \
                    / np.sum(np.abs(obj_est_n)**2)
            return np.sum(np.abs(obj - g_fac * obj_est_n)**2) \
                / np.sum(np.abs(obj)**2)

        RMS = np.array(list(map(gamma, obj_est_n)))

        return obj_est, RMS, R_factor

    def repie(self, obj, diff_patterns, **kwargs):
        """
        Args:
            obj: Object
            diff_patterns: Diffraction pattern
            **kwargs: Arbitrary keyword arguments.
        Returns:
            Estimated object, root mean square error and sum of least squares error
            of the estimate
        """

        # parameters
        hold = kwargs.get('hold', 1)
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

        # illumination function initial guess
        illu_func_est = np.ones((self._probe, self._probe))

        # initialization for R_factor
        R_factor = np.zeros(err_n)

        # holder variable for the estimated object after some iterations
        obj_est_n = np.zeros((err_n, height, width), dtype=np.complex64)

        cmask = circ_mask(self._probe, (self._probe//2, self._probe//2),
            self._probe//2, 1.0)

        diff_pat_sum = np.sum(np.abs(diff_patterns[half_rc])**2)
        MN = np.prod(diff_patterns[half_rc].shape)

        k = 0

        while k < self._iterations:
            ext_waves = []
            ext_wave_diffs = []

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
                ext_diff_c = diff_patterns[x_loc * self._rc[0] + y_loc] * \
                    np.exp(1j * np.angle(ext_diff_g))
                ext_wave_c = ifft2(ifftshift(ext_diff_c))

                if k >= hold:
                    # probe power correction
                    illu_func_est = illu_func_est * np.sqrt(diff_pat_sum / (MN * np.sum(np.abs(illu_func_est)**2)))

                ext_wave_upd = obj_g + (ext_wave_c - ext_wave_g) \
                    * illu_func_est.conj() \
                    / ((1 - self._alpha) * np.max(np.abs(illu_func_est))**2 \
                    + self._alpha * np.max(np.abs(illu_func_est))**2)
                obj_est[y_i:y_i+self._probe, x_i:x_i+self._probe] = ext_wave_upd

                ext_wave_diffs.append(diff_patterns[x_loc * self.rc[0] + y_loc])
                ext_waves.append(ext_diff_g)

                if k >= hold:
                    # with probe centering
                    illu_func_est = illu_func_est + ((ext_wave_c - ext_wave_g) \
                    * obj_g_cpy.conj()  - cmask * illu_func_est) \
                    / ((1 - self._beta) * np.max(np.abs(obj_g_cpy))**2 \
                    + self._beta * np.max(np.abs(obj_g_cpy))**2 + cmask)

            if k % err_ival == 0:
                ext_waves = np.array(ext_waves)
                ext_wave_diffs = np.array(ext_wave_diffs)
                numerator = np.abs(ext_wave_diffs - np.abs(ext_waves))
                denominator = np.abs(ext_wave_diffs)
                R_factor[err_i] = np.sum(np.sum(numerator)) / np.sum(np.sum(denominator))
                obj_est_n[err_i] = obj_est
                err_i += 1

            k += 1

        def gamma(obj_est_n):
            g_fac = np.sum(obj * obj_est_n.conj()) \
                    / np.sum(np.abs(obj_est_n)**2)
            return np.sum(np.abs(obj - g_fac * obj_est_n)**2) \
                / np.sum(np.abs(obj)**2)

        RMS = np.array(list(map(gamma, obj_est_n)))

        return obj_est, illu_func_est, RMS, R_factor
