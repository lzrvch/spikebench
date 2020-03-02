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

from functools import partial
from sklearn.base import TransformerMixin

from pyspikelib.base_transformers import NoFitMixin


class SpikeTrainTransform(TransformerMixin, NoFitMixin):
    """Base class for spike train transforms"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def string_to_float_series(string_series, delimiter=None):
        return np.array([float(value) for value in string_series.split(delimiter)])

    @staticmethod
    def numpy_transform(tensor, axis):
        raise NotImplementedError

    @staticmethod
    def single_train_transform(tensor):
        raise NotImplementedError

    def transform(self, X, y=None, format='numpy', axis=-1, delimiter=None):
        if format == 'numpy':
            X = self.numpy_transform(X, axis=axis)
            return X
        join_delimiter = ' ' if delimiter is None else delimiter
        for train_index, spike_train in enumerate(X.series.values):
            train = self.string_to_float_series(spike_train, delimiter=delimiter)
            transfomed_train = self.single_train_transform(train)
            transfomed_series = join_delimiter.join(
                ['{:.2f}'.format(value) for value in transfomed_train]
            )
            X.series.iloc[train_index] = transfomed_series
        return X


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

    @staticmethod
    def numpy_transform(tensor, axis=-1):
        self.shuffle_along_axis(tensor, axis)
        return tensor

    @staticmethod
    def single_train_transform(tensor):
        np.random.shuffle(tensor)
        return tensor


class TrainBinarizationTransform(SpikeTrainTransform):
    """Turn an ISI series into a sequence of time bins with spike occurences"""

    def __init__(self, bin_size, keep_spike_counts=True, train_duration=None):
        super().__init__()
        self.bin_size = bin_size
        self.keep_counts = keep_spike_counts
        self.train_duration = train_duration

    @staticmethod
    def numpy_transform(tensor, axis=0):
        saxis = [
            samples_axis for samples_axis in range(tensor.ndim) if samples_axis != axis
        ]
        return np.apply_along_axis(
            partial(self.single_train_transform, axis=axis), saxis, tensor
        )

    @staticmethod
    def single_train_transform(series, axis):
        spike_times = np.cumsum(series, axis)
        duration = (
            spike_times.max() if self.train_duration is None else self.train_duration
        )
        return np.histogram(
            spike_times,
            bins=int(duration / self.bin_size),
            range=(spike_times.min(), duration),
        )[0].astype(np.float32)


# ToDo: jitter, rate estimation ...
