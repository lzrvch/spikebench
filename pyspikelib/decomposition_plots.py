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

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from .mpladeq import beautify_mpl, prettify


def feature_scatter_plot(
    X,
    y,
    features,
    samples=1000,
    legend=None,
    xaxis=None,
    yaxis=None,
    figsize=(15, 8),
    alpha=0.3,
):
    sns.set(palette='Set2', style='ticks', font_scale=1.7)

    indices = np.random.choice(X.shape[0], samples)

    colors = pd.Series(y[indices]).map(
        {0: sns.color_palette('Paired')[5], 1: sns.color_palette('Paired')[1]}
    )

    custom_lines = [
        Line2D([0], [0], color=sns.color_palette('Paired')[5], lw=1.5),
        Line2D([0], [0], color=sns.color_palette('Paired')[1], lw=1.5),
    ]

    beautify_mpl()

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        X.loc[:, features[0]].values[indices],
        X.loc[:, features[1]].values[indices],
        c=colors,
        alpha=alpha,
    )

    prettify()
    if legend:
        ax.legend(custom_lines, legend)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)

    return ax


def decompose_scatter_plot(
    X,
    y,
    features,
    reducer,
    samples=1000,
    legend=None,
    xaxis=None,
    yaxis=None,
    supervised=False,
    figsize=(15, 8),
    alpha=0.3,
):
    sns.set(palette='Set2', style='ticks', font_scale=1.7)

    indices = np.random.choice(X.shape[0], samples)

    colors = pd.Series(y[indices]).map(
        {0: sns.color_palette('Paired')[5], 1: sns.color_palette('Paired')[1]}
    )

    custom_lines = [
        Line2D([0], [0], color=sns.color_palette('Paired')[5], lw=1.5),
        Line2D([0], [0], color=sns.color_palette('Paired')[1], lw=1.5),
    ]

    beautify_mpl()

    fig, ax = plt.subplots(figsize=figsize)

    if supervised:
        train_indices = list(set(range(X.shape[0])) - set(indices))
        mapper = reducer[0](**reducer[1]).fit(
            X.loc[:, features].values[train_indices, :], y[train_indices]
        )
        X2d = mapper.transform(X.loc[:, features].values[indices, :])
    else:
        X2d = reducer[0](**reducer[1]).fit_transform(
            X.loc[:, features].values[indices, :]
        )

    ax.scatter(X2d[:, 0], X2d[:, 1], c=colors, alpha=alpha)

    prettify()
    if legend:
        ax.legend(custom_lines, legend)

    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.xticks([])
    plt.yticks([])

    return ax
