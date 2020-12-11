from tqdm import tqdm
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial

import elephant.statistics as spkstat
import numpy as np
from elephant.kernels import GaussianKernel
from neo.core import SpikeTrain
from quantities import ms
from sklearn.base import TransformerMixin

from pyspikelib.base_transformers import NoFitMixin


class SpikeTrainTransform(TransformerMixin, NoFitMixin):
    """Base class for spike train transforms"""

    def __init__(self, n_jobs=None):
        super().__init__()
        self.n_jobs = mp.cpu_count() if n_jobs is None else n_jobs

    @staticmethod
    def string_to_float_series(string_series, delimiter=None):
        return np.array([float(value) for value in string_series.split(delimiter)])

    def numpy_transform(self, tensor, axis):
        raise NotImplementedError

    def single_train_transform(self, tensor):
        raise NotImplementedError

    def transform(self, X, y=None, format='pndz', axis=-1, delimiter=None):
        if format == 'numpy':
            X = self.numpy_transform(X, axis)
            return X
        join_delimiter = ' ' if delimiter is None else delimiter

        def transform_spike_train(spike_train):
            train = self.string_to_float_series(spike_train, delimiter=delimiter)
            transfomed_train = self.single_train_transform(train)
            return join_delimiter.join(
                ['{:.2f}'.format(value) for value in transfomed_train]
            )

        pool = Pool(self.n_jobs)
        X.series = pool.map(transform_spike_train, X.series.values)
        return X


class DFSpikeTrainTransform(SpikeTrainTransform):

    def __init__(self, func, n_jobs=None, copy=False, **kwargs):
        super().__init__()
        self.func = func
        self.copy = copy
        self.kwargs = kwargs

    def set_params(self, **params):
        for key, value in params.items():
            self.kwargs[key] = value

    def single_train_transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_, **self.kwargs)


class ISIShuffleTransform(SpikeTrainTransform):
    """Randomly permute ISIs in each spike train in-place"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def shuffle_along_axis(tensor, axis):
        b = tensor.swapaxes(axis, -1)
        shp = b.shape[:-1]
        for ndx in np.ndindex(shp):
            np.random.shuffle(b[ndx])

    def numpy_transform(self, tensor, axis=-1):
        self.shuffle_along_axis(tensor, axis)
        return tensor

    def single_train_transform(self, tensor):
        np.random.shuffle(tensor)
        return tensor


class TrainBinarizationTransform(SpikeTrainTransform):
    """Turn an ISI series into a sequence of time bins with spike occurences"""

    def __init__(
        self, bin_size, keep_spike_counts=True, train_duration=None, start_time=None
    ):
        super().__init__()
        self.bin_size = bin_size
        self.keep_counts = keep_spike_counts
        self.train_duration = train_duration
        self.start_time = start_time
        self.fixed_size_output = (
            self.start_time is not None and self.train_duration is not None
        )

    def numpy_transform(self, tensor, axis=1):
        if self.fixed_size_output:
            return np.apply_along_axis(
                partial(self.single_train_transform, axis=axis), axis, tensor
            )
        else:
            single_train_transform = partial(self.single_train_transform, axis=axis)

            def variable_length_transform(series):
                return list(single_train_transform(series))

            return np.apply_along_axis(variable_length_transform, axis, tensor)

    def single_train_transform(self, series, axis):
        spike_times = np.cumsum(series, axis)
        start_time = spike_times.min() if self.start_time is None else self.start_time
        duration = (
            spike_times.max()
            if self.train_duration is None
            else self.train_duration + start_time
        )
        binarized_spike_train = np.histogram(
            spike_times,
            bins=int(duration / self.bin_size),
            range=(start_time, duration),
        )[0].astype(np.float32)
        if not self.keep_counts:
            binarized_spike_train = (binarized_spike_train > 0).astype(np.float32)
        return binarized_spike_train


class SpikeTimesToISITransform(SpikeTrainTransform):
    def __init__(self):
        super().__init__()

    def numpy_transform(self, tensor, axis=1):
        return np.diff(tensor, axis=axis)

    def single_train_transform(self, tensor):
        return np.diff(tensor)


class ISIToSpikeTimesTransform(SpikeTrainTransform):
    def __init__(self):
        super().__init__()

    def numpy_transform(self, tensor, axis=1):
        return np.cumsum(tensor, axis=axis)

    def single_train_transform(self, tensor):
        return np.cumsum(tensor)


class SpikeTrainToFiringRateTransform(SpikeTrainTransform):
    def __init__(
        self,
        kernel_width=None,
        isi_input=True,
        start_time=None,
        train_duration=None,
        sampling_period=None,
    ):
        super().__init__()
        self.kernel_width = kernel_width
        self.isi_input = isi_input
        self.start_time = start_time
        self.train_duration = train_duration
        self.sampling_period = sampling_period
        self.fixed_size_output = (
            self.start_time is not None and self.train_duration is not None
        )

    @staticmethod
    def get_rate_estimate(
        spike_times, kernel_width, start_time, train_duration, sampling_period
    ):
        t_start = spike_times[0] if start_time is None else start_time
        t_stop = spike_times[-1] if train_duration is None else t_start + train_duration
        train = SpikeTrain(
            spike_times[(spike_times > t_start) & (spike_times < t_stop)] * ms,
            t_start=t_start,
            t_stop=t_stop,
        )
        kernel = GaussianKernel(sigma=kernel_width * ms)
        rate = spkstat.instantaneous_rate(
            train, kernel=kernel, sampling_period=sampling_period * ms
        )
        return np.array(rate)[:, 0]

    def numpy_transform(self, tensor, axis=1):
        if self.fixed_size_output:
            return np.apply_along_axis(
                partial(self.single_train_transform, axis=axis), axis, tensor
            )
        else:
            single_train_transform = partial(self.single_train_transform, axis=axis)

            def variable_length_transform(series):
                return ' '.join(
                    [
                        '{:2f}'.format(value)
                        for value in list(single_train_transform(series))
                    ]
                )

            return np.apply_along_axis(variable_length_transform, axis, tensor)

    def single_train_transform(self, tensor, axis=-1):
        if self.isi_input:
            spike_times = np.cumsum(tensor, axis=axis)
        return self.get_rate_estimate(
            spike_times,
            self.kernel_width,
            self.start_time,
            self.train_duration,
            self.sampling_period,
        )


# ToDo: jitter transform
