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
            raise ValueError('window must be one of "flat", "hanning",'
                              '"hamming", "bartlett", "blackman"')

        s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        if window == 'flat': # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    @staticmethod
    def filter(x, fps):
        fc = 1  # cut-off frequency of the filter
        w = fc / (fps / 2) # normalize the frequency
        b, a = signal.butter(5, w, 'low')
        return signal.filtfilt(b, a, x)

    def numpy_transform(self, tensor, axis=1):
        return np.apply_along_axis(
            self.single_train_transform, axis, tensor
        )
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
        d, trace = oopsi.fast(x, dt=dt_factor/fps, iter_max=max_iters)
        return trace


    def numpy_transform(self, tensor, axis=1):
        return np.apply_along_axis(
            self.single_train_transform, axis, tensor
        )
        return tensor

    def single_train_transform(self, tensor):
        return self.apply_oopsi(tensor, self.fps, self.dt_factor, self.max_iters)
