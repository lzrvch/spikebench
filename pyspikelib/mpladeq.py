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

import matplotlib as mpl
import matplotlib.pylab as plt
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
    plt.style.use('seaborn-pastel')
    sns.set_color_codes()


def prettify(figsize=(8, 6)):
    ax = plt.gca()
    ma(ax, nbins=5)
    plt.gcf().autofmt_xdate()
    mpl.rcParams['figure.figsize'] = figsize


def boxplot(data, x, y, figsize=(10, 8), xticklabels=None, savefile=None):
    fig, ax = plt.subplots(1)
    sns.boxplot(x=x, y=y, data=data, notch=True, linewidth=2, width=0.4)
    prettify(figsize)

    for i, artist in enumerate(ax.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        # col = artist.get_facecolor()
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
