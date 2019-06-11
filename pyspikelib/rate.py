import numpy as np
import elephant.statistics as spkstat

import multiprocessing as mp

from tqdm import tqdm
from quantities import ms
from neo.core import SpikeTrain
from elephant.kernels import GaussianKernel


class RateEstimator():

    def __init__(self, width=None, bin_size=None):

        self.width = width
        self.bin_size = bin_size
        self.binary = False

    def rate(self, series):

        return firing_rate_estimate(series,
                                    width=self.width)

    def binarize(self, series):

        return binarize_series(series,
                               bin_size=self.bin_size,
                               binary=self.binary)


def firing_rate_estimate(isi_series, width=100):

    spikes = np.cumsum(isi_series)
    train = SpikeTrain(spikes * ms, t_stop=spikes[-1])
    kernel = GaussianKernel(sigma=width*ms)
    rate = spkstat.instantaneous_rate(train,
                                      kernel=kernel,
                                      sampling_period=ms)
    return np.array(rate)[:, 0]


def binarize_series(isi_series, bin_size=10, binary=False):

    spikes = np.cumsum(isi_series)
    binary_series = np.zeros(shape=int(spikes[-1] / bin_size) + 1)
    for spike_time in spikes:
        if not binary:
            binary_series[int(spike_time / bin_size)] += 1
        else:
            binary_series[int(spike_time / bin_size)] = 1

    return binary_series


def convert_isi_series_to_rate(isi_series, width=100,
                               progressbar=False, parallel=True):

    rate_series = {}
    rate_series['series'] = []

    if progressbar:
        pbar = tqdm(total=len(isi_series['series']))

    if parallel:
        estimator = RateEstimator(width=width)
        pool = mp.Pool(mp.cpu_count())
        rate_series['series'] = pool.map(estimator.rate, isi_series['series'])
    else:
        for series in isi_series['series']:
            rate_signal = firing_rate_estimate(series, width=width)
            rate_series['series'].append(rate_signal)
            if progressbar:
                pbar.update()

    rate_series['ids'] = isi_series['ids']

    return rate_series


def binarize_spike_train(isi_series, bin_size=10, binary=False):

    binary_series = {}
    binary_series['series'] = []

    estimator = RateEstimator(bin_size=bin_size)
    estimator.binary = binary
    pool = mp.Pool(mp.cpu_count())
    binary_series['series'] = pool.map(estimator.binarize, isi_series['series'])
    binary_series['ids'] = isi_series['ids']

    return binary_series
