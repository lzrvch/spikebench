import numpy as np
from sklearn.base import TransformerMixin

from pyspikelib.base_transformers import NoFitMixin


class ISIShuffleTransform(TransformerMixin, NoFitMixin):
    def __init__(self):
        super().__init__(self)

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
            shuffled_series = delimiter.join(
                ['{:.2f}'.format(value) for value in np.random.shuffle(train)]
            )
            X.series.iloc[train_index, :] = shuffled_series
        return X


# ToDo: binarization, jitter, rate estimation ...
