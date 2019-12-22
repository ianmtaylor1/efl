# -*- coding: utf-8 -*-
"""
util.py

Utility functions for the efl package that are used multiple places but don't
fit into any particular sub-package.
"""

import matplotlib.pyplot as plt
import matplotlib
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


# The following functions were -stolen- borrowed from
# https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
# And then modified to fit my use case. Some additional code was taken from
# https://matplotlib.org/3.1.1/gallery/axes_grid1/demo_axes_divider.html


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(numpy.arange(data.shape[1]))
    ax.set_yticks(numpy.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(numpy.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(numpy.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, numpy.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts