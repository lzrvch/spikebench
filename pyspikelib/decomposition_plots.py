import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd

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
