# -*- coding: utf-8 -*-
"""
util.py

Utility functions for the efl package that are used multiple places but don't
fit into any particular sub-package.
"""

import matplotlib.pyplot as plt
import numpy
from scipy.stats import kde

def make_axes(num_plots, nrows, ncols, figsize, main_title):
    """Create figures and plots to be used by plotting functions that need to
    plot multiple things, possible more than one to a figure.
    Parameters:
        num_plots - number of plots that will be plotted. One Axes will be
            created for each.
        nrows, ncols - the number of rows and columns to use to arrange the
            axes in each figure. See the documentation of
            matplotlib.pyplot.subplots
        figsize - figure size, passed to the matplotlib.pyplot.subplots
            command. See matplotlib.pyplot.subplots documentation.
        main_title - the title of each figure created. If None, no title will
            be added to the plot. Else, a title in the format
            'main_title #/#' will be added indicating the number of the plot
            in the sequence
    Returns: a generator that yields figures one at a time. Each figure except
        the last will have axes in a layout determined by nrows and ncols. The
        last figure will have fewer if num_plots is not divisible by
        nrows*ncols
    """
    numfigs = (num_plots // (nrows*ncols)) + ((num_plots % (nrows*ncols)) > 0)
    remainder = num_plots % (nrows*ncols)
    for i in range(numfigs):
        fig, _ = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                                 constrained_layout=True, squeeze=False)
        if main_title is not None:
            fig.suptitle("{} {}/{}".format(main_title, i+1, numfigs))
        # If we've made too many axes, delete the ones we don't need
        if (i == numfigs-1) and (remainder > 0):
            for ax in fig.axes[remainder:]:
                fig.delaxes(ax)
        yield fig


def draw_densplot(ax, data, nout=220, scale_height=1.0):
    """Draw a density plot on the axes provided.
    Parameters:
        ax - the axes to draw on
        data - the data to estimate the density with
        nout - number of points to use when drawing the density
    """
    density = kde.gaussian_kde(data)
    xlow = min(data) - 0.05*(max(data) - min(data))
    xhigh = max(data) + 0.05*(max(data) - min(data))
    x = numpy.arange(xlow, xhigh, (xhigh-xlow)/nout)
    y = density(x) * scale_height
    ax.plot(x,y)