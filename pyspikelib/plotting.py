import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
import seaborn as sns


def make_up_axis(ax):
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def ma(ax, nbins=5):
    make_up_axis(ax)
    return ax


def un_frame_axis(ax):
    for o in ['right', 'top', 'left', 'bottom']:
        ax.spines[o].set_visible(False)


def frame_axis(ax):
    for o in ['right', 'top', 'left', 'bottom']:
        ax.spines[o].set_visible(True)
        ax.spines[o].set_color('#333333')


def params_make_fs(fs, figsize):
    return {
        'axes.titlesize': fs,
        'axes.labelsize': fs,
        'font.size': fs,
        'legend.fontsize': fs,
        'xtick.labelsize': fs,
        'ytick.labelsize': fs,
        'text.usetex': False,
        'figure.figsize': figsize,
    }


def beautify_mpl(fontsize=16, figsize=(10, 8), dark_mode=False):
    sns.set(font_scale=1.7, style='ticks')
    params = params_make_fs(fontsize, figsize)
    mpl.rcParams['axes.facecolor'] = '#ffffff'
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['savefig.transparent'] = True
    mpl.rcParams['axes.labelcolor'] = '#000000'
    mpl.rcParams['xtick.color'] = '#000000'
    mpl.rcParams['ytick.color'] = '#000000'
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams.update(params)
    if dark_mode:
        plt.style.use('dark_background')
    plt.style.use('seaborn-deep')
    sns.set_color_codes()


def prettify(figsize=(8, 6)):
    ax = plt.gca()
    ma(ax, nbins=5)
    plt.gcf().autofmt_xdate()
    mpl.rcParams['figure.figsize'] = figsize


def boxplot(data, x, y, figsize=(10, 8), xticklabels=None,
            savefile=None, box_quantiles=None, notch=False):
    fig, ax = plt.subplots(1)
    whis = (1, 99) if box_quantiles is not None else 1.5
    sns.boxplot(x=x, y=y, data=data, notch=notch,
                linewidth=2, width=0.4, whis=whis)
    prettify(figsize)

    for i, artist in enumerate(ax.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        # col = artist.get_facecolor()
        col = (0.0, 0.0, 0.0)
        artist.set_edgecolor((0.0, 0.0, 0.0))
        artist.set_facecolor('None')

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)

    plt.xlabel('')
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')

    if savefile is not None:
        plt.savefig(savefile, format='eps', bbox_inches='tight')


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

    cmap = ('#fb290f', '#063373')
    # cmap = (sns.color_palette('Paired')[5], sns.color_palette('Paired')[1])

    colors = pd.Series(y[indices]).map(
        {0: cmap[0], 1: cmap[1]}
    )

    custom_lines = [
        Line2D([0], [0], color=cmap[0], lw=1.5),
        Line2D([0], [0], color=cmap[1], lw=1.5),
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


def embedding_scatter_plot(
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

    cmap = ('#fb290f', '#063373')

    colors = pd.Series(y[indices]).map(
        {0: cmap[0], 1: cmap[1]}
    )

    custom_lines = [
        Line2D([0], [0], color=cmap[0], lw=1.5),
        Line2D([0], [0], color=cmap[1], lw=1.5),
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
