import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np


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


def beautify_mpl(fontsize=16, figsize=(6, 5)):
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


def prettify(figsize=(8, 6)):
    ax = plt.gca()
    ma(ax, nbins=5)
    plt.gcf().autofmt_xdate()
    mpl.rcParams['figure.figsize'] = figsize
