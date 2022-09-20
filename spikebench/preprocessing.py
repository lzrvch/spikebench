import numpy as np
import pandas as pd
import scipy.signal as signal
from pathos.multiprocessing import ProcessingPool as Pool

import spikebench.oopsi as oopsi
from spikebench import SpikeTrainTransform


class SmoothingTransform(SpikeTrainTransform):
    """Smooth a set of time series"""

    def __init__(self, window_len=7, window='hanning', do_filtering=False, fps=None):
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
            return x
            # raise ValueError('Input vector needs to be bigger than window size.')

        if window_len < 3:
            return x

        if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError(
                'window must be one of "flat", "hanning",'
                '"hamming", "bartlett", "blackman"'
            )

        s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
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

    def __init__(self, dt_factor=1e-1, max_iters=100, timespan=None):
        super().__init__()
        self.dt_factor = dt_factor
        self.max_iters = max_iters
        self.timespan = timespan

    @staticmethod
    def apply_oopsi(x, fps, dt_factor, max_iters):
        if fps is not None:
            events, trace = oopsi.fast(x, dt=dt_factor / fps, iter_max=max_iters)
        else:
            raise RuntimeError('Invalid fps value!')
        return events, trace

    def set_params(self, timespan):
        self.timespan = timespan

    def numpy_transform(self, tensor, axis=1):
        self.fps = tensor.shape[1] / self.timespan
        return np.apply_along_axis(self.single_train_transform, axis, tensor)

    def single_train_transform(self, tensor, timespan=None):
        self.fps = tensor.shape[0] / timespan
        return self.apply_oopsi(tensor, self.fps, self.dt_factor, self.max_iters)

    def transform(self, X, y=None, axis=-1, delimiter=None):
        join_delimiter = ' ' if delimiter is None else delimiter

        def transform_spike_train(spike_train):
            spike_train, timespan = spike_train
            train = self.string_to_float_series(spike_train, delimiter=delimiter)
            events, transfomed_train = self.single_train_transform(
                train, timespan=timespan
            )
            transformed_train_string = join_delimiter.join(
                ['{:.5f}'.format(value) for value in transfomed_train]
            )
            events_string = join_delimiter.join(
                ['{:.5f}'.format(value) for value in events]
            )
            return transformed_train_string, events_string

        pool = Pool(self.n_jobs)
        X.series = pool.map(
            transform_spike_train,
            [
                (series, timespan)
                for series, timespan in zip(X.series.values, X.timespan.values)
            ],
        )
        return X


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

    def __init__(self, window=2001, peak_std_threshold=4.0, n_jobs=None):
        super().__init__(n_jobs=n_jobs)
        self.window = window
        self.peak_std_threshold = peak_std_threshold
        self.mad_constant = 1.4826

    def _robust_std(self, x):
        '''Robust estimate of std'''
        MAD = np.median(np.abs(x - np.median(x)))
        return self.mad_constant * MAD

    def detrend_trace(self, trace):
        trend_component = signal.medfilt(trace, self.window)
        trend_component = np.minimum(
            trend_component, self.peak_std_threshold * self._robust_std(trend_component)
        )
        return trace - trend_component

    def numpy_transform(self, tensor, axis=1):
        return np.apply_along_axis(self.single_train_transform, axis, tensor)

    def single_train_transform(self, tensor):
        return self.detrend_trace(tensor)


class Normalize(SpikeTrainTransform):
    def __init__(self, window=1000, percentile=8):
        super().__init__()
        self.window = window
        self.percentile = percentile

    @staticmethod
    def normalize_trace(trace, window=1000, percentile=8):
        lower_percentile = lambda x: np.percentile(x, percentile)
        baseline = (
            pd.Series(trace)
            .rolling(window=window, center=True)
            .apply(func=lower_percentile)
        )
        baseline = baseline.fillna(method='bfill')
        baseline = baseline.fillna(method='ffill')
        dF = trace - baseline
        dFF = dF / baseline
        return dFF

    def numpy_transform(self, tensor, axis=0):
        return np.apply_along_axis(self.single_train_transform, axis, tensor)

    def single_train_transform(self, tensor):
        return self.normalize_trace(tensor, self.window, self.percentile)
