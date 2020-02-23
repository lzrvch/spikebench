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
from sklearn.base import TransformerMixin

from pyspikelib.base_transformers import NoFitMixin


class ISIShuffleTransform(TransformerMixin, NoFitMixin):
    """Randomly permute ISIs in each spike train in-place"""
    def __init__(self):
        super().__init__()

    @staticmethod
    def shuffle_along_axis(tensor, axis=-1):
        b = tensor.swapaxes(axis, -1)
        shp = b.shape[:-1]
        for ndx in np.ndindex(shp):
            np.random.shuffle(b[ndx])

    @staticmethod
    def string_to_float_series(string_series, delimiter=None):
        return np.array([float(value) for value in string_series.split(delimiter)])

    def transform(self, X, y=None, format='numpy', axis=-1, delimiter=None):
        if format == 'numpy':
            self.shuffle_along_axis(X, axis=axis)
            return X
        for train_index, spike_train in enumerate(X.series.values):
            train = self.string_to_float_series(spike_train, delimiter=delimiter)
            np.random.shuffle(train)
            shuffled_series = delimiter.join(
                ['{:.2f}'.format(value) for value in train]
            )
            X.series.iloc[train_index] = shuffled_series
        return X


# ToDo: binarization, jitter, rate estimation ...
