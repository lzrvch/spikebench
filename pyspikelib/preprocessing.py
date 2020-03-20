# PySpikeLib: A set of tools for neuronal spiking data mining
# Copyright (c) 2020 Ivan Lazarevich.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import scipy.signal as signal

import pyspikelib.oopsi as oopsi
from pyspikelib import SpikeTrainTransform


class SmoothingTransform(SpikeTrainTransform):
    """Smooth a set of time series"""

    def __init__(self, window_len=10, window='hanning', do_filtering=False, fps=None):
        super().__init__()
        self.window_len = window_len
        self.window = window
        self.do_filtering = do_filtering
        self.fps = fps

    @staticmethod
    def smooth(x, window_len, window='hanning'):
        if x.ndim != 1:
            raise ValueError('smooth() method only accepts 1 dimensional arrays')

        if x.size < window_len:
            raise ValueError('Input vector needs to be bigger than window size.')

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError(
                'window must be one of "flat", "hanning",'
                '"hamming", "bartlett", "blackman"'
            )

        s = np.r_[x[window_len - 1: 0: -1], x, x[-2: -window_len - 1: -1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    @staticmethod
    def filter(x, fps):
        fc = 1  # cut-off frequency of the filter
        w = fc / (fps / 2)  # normalize the frequency
        b, a = signal.butter(5, w, 'low')
        return signal.filtfilt(b, a, x)

    def numpy_transform(self, tensor, axis=1):
        return np.apply_along_axis(self.single_train_transform, axis, tensor)
        return tensor

    def single_train_transform(self, tensor):
        tensor = self.smooth(tensor, self.window_len, self.window)
        output = self.filter(tensor, self.fps) if self.do_filtering else tensor
        return output


class OOPSITransform(SpikeTrainTransform):
    """Apply OOPSI algorithm to traces"""

    def __init__(self, dt_factor=1e-1, max_iters=500, fps=None):
        super().__init__()
        self.dt_factor = dt_factor
        self.max_iters = max_iters
        self.fps = fps

    @staticmethod
    def apply_oopsi(x, fps, dt_factor, max_iters):
        d, trace = oopsi.fast(x, dt=dt_factor / fps, iter_max=max_iters)
        return trace

    def numpy_transform(self, tensor, axis=1):
        return np.apply_along_axis(self.single_train_transform, axis, tensor)
        return tensor

    def single_train_transform(self, tensor):
        return self.apply_oopsi(tensor, self.fps, self.dt_factor, self.max_iters)


class MedianFilterDetrender(SpikeTrainTransform):
    """
    Modified from: https://github.com/AllenInstitute/neuroglia/
    Detrend the calcium signal using the local median
    Parameters
    ----------
    window : int, optional (default: 101)
        Number of samples to use to compute local median
    peak_std_threshold : float, optional (default: 4.0)
        If the median exceeds this threshold, it will be capped at this level.
    """

    def __init__(self, window=101, peak_std_threshold=4.0):
        self.window = window
        self.peak_std_threshold = peak_std_threshold
        self.mad_constant = 1.4826

    @staticmethod
    def _robust_std(x):
        '''Robust estimate of std
        '''
        MAD = np.median(np.abs(x - np.median(x)))
        return self.mad_constant * MAD

    def core_transform(self, X):
        self.fit_params = {}
        X_new = X.copy()
        for col in X.columns:
            tmp_data = X[col].values.astype(np.double)
            mf = medfilt(tmp_data, self.window)
            mf = np.minimum(mf, self.peak_std_threshold * self._robust_std(mf))
            self.fit_params[col] = dict(mf=mf)
            X_new[col] = tmp_data - mf

        return X_new

    def numpy_transform(self, tensor, axis=1):
        transformed_df = self.core_transform(pd.DataFrame(tensor))
        return transformed_df.values

    def single_train_transform(self, tensor):
        transformed_df = self.core_transform(pd.DataFrame(tensor))
        return transformed_df.values


class Normalize(SpikeTrainTransform):
    """
    Modified from: https://github.com/AllenInstitute/neuroglia/
    Normalize the trace by a rolling baseline (that is, calculate dF/F)
    Parameters
    ---------
    window: float, optional (default: 3.0)
        time in minutes
    percentile: int, optional (default: 8)
        percentile to subtract off
    """

    def __init__(self, window=3.0, percentile=8):
        self.window = window
        self.percentile = percentile

    def transform(self, X):
        """Normalize each column of X
        Parameters
        ----------
        X : DataFrame in `traces` structure [n_samples, n_traces]
        Returns
        -------
        Xt : DataFrame in `traces` structure [n_samples, n_traces]
            The normalized calcium traces.
        """
        df_norm = pd.DataFrame()
        for col in X.columns:
            df_norm[col] = self.normalize_trace(
                trace=X[col], window=self.window, percentile=self.percentile,
            )

        return df_norm

    @staticmethod
    def normalize_trace(trace, window=3, percentile=8):
        """ normalized the trace by substracting off a rolling baseline
        Parameters
        ---------
        trace: pd.Series with time as index
        window: float
            time in minutes
        percentile: int
            percentile to subtract off
        """

        sampling_rate = np.diff(trace.index).mean()
        window = int(np.ceil(window / sampling_rate))

        # suggest 8% in literature, but this doesnt work well for our data, use median
        p = lambda x: np.percentile(x, percentile)
        baseline = trace.rolling(window=window, center=True).apply(func=p)
        baseline = baseline.fillna(method='bfill')
        baseline = baseline.fillna(method='ffill')
        dF = trace - baseline
        dFF = dF / baseline

        return dFF

    def numpy_transform(self, tensor, axis=1):
        transformed_df = self.core_transform(pd.DataFrame(tensor))
        return transformed_df.values

    def single_train_transform(self, tensor):
        transformed_df = self.core_transform(pd.DataFrame(tensor))
        return transformed_df.values
