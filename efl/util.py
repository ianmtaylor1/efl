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
    plot multiple things, possibly more than one to a figure.
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


def draw_densplot(ax, data, nout=220, scale_height=1.0, **kwargs):
    """Draw a density plot on the axes provided.
    Parameters:
        ax - the axes to draw on
        data - the data to estimate the density with
        nout - number of points to use when drawing the density
        scale_height - scale the height of the density by this amount
        **kwargs - passed to ax.plot()
    """
    density = kde.gaussian_kde(data, bw_method="scott")
    xlow = min(data) - 0.05*(max(data) - min(data))
    xhigh = max(data) + 0.05*(max(data) - min(data))
    x = numpy.arange(xlow, xhigh, (xhigh-xlow)/nout)
    y = density(x) * scale_height
    return ax.plot(x, y, **kwargs)


# The following function was -stolen- borrowed from
# https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
# And then modified to fit my use case.


def heatmap(data, ax, row_labels=None, col_labels=None,
            include_cbar=True, cbarlabel="", annotate=True, maxtextlen=40,
            valfmt='{x:.0%}', textcolors=["white", "black"], **kwargs):
    """Create a heatmap from a pandas DataFrame (most likely from pivot_table)
    
    Parameters:
        data - A NxM pandas DataFrame containing the data to plot
        ax - matplotlib.axes.Axes on which to plot
        row_labels - a list or array of length N with the labels for the rows.
            If None, the index of data is used.
        col_labels - a list or array of length M with the labels for the
            columns. If None, the columns of data are used.
        include_cbar - A bool. If true, include a colorbar legend. If false,
            don't include a colorbar legend.
        cbarlabel - The label for the colorbar.  Optional.
        annotate - A bool. If true, annotate each square with its value.
        maxtextlen - if the total length of the x-labels exceeds this value,
            they will be rotated to avoid overlap.
        valfmt - A format string to use formatting both the annotated values
            (if any) and the colorbar ticks (if any)
        textcolors - A length-2 list-like with color names for the 
            annotations on low and high values, respectively.
        **kwargs - any remaining arguments passed to imshow()
    
    Returns: the Axes object passed in ax.
    """
    # Default values for row and column labels
    if row_labels is None:
        row_labels = list(data.index)
    if col_labels is None:
        col_labels = list(data.columns)
    # Give ourselves a numpy array to work with
    data_array = numpy.array(data)
    
    # Plot the heatmap
    im = ax.imshow(data_array, origin="lower", **kwargs)
    
    # Make the formatter for colorbar labels and annotations
    formatter = matplotlib.ticker.StrMethodFormatter(valfmt)
    
    # Make the colorbar, if required
    if include_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, format=formatter)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    
    # We want to show all ticks...
    ax.set_xticks(numpy.arange(data_array.shape[1]))
    ax.set_yticks(numpy.arange(data_array.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)
    
    # Rotate the tick labels and set their alignment.
    total_xlab_len = sum(len(str(lb)) for lb in col_labels)
    if total_xlab_len > maxtextlen:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                 rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    
    # Draw ticks
    ax.set_xticks(numpy.arange(data_array.shape[1]+1)-.5, minor=True)
    ax.set_yticks(numpy.arange(data_array.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Annotate, if required
    if annotate:
        # Normalize the threshold to the images color range.
        threshold = im.norm(data_array.max())/2
        # Annotate each cell
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                textcolor = textcolors[int(im.norm(data_array[i, j]) > threshold)]
                im.axes.text(j, i, formatter(data_array[i, j], None),
                             color=textcolor, horizontalalignment="center", 
                             verticalalignment="center")
    
    # Return the axes we just drew on
    return ax