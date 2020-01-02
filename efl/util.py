# -*- coding: utf-8 -*-
"""
util.py

Utility functions for the efl package that are used multiple places but don't
fit into any particular sub-package.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy
import pandas
from scipy.stats import kde


###########################################################
## PLOTTING HELPER FUNCTIONS ##############################
###########################################################

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
    return im


###########################################################
## FUNCTIONS FOR CREATING "OTHER" CATEGORIES IN SUMMARIES #
###########################################################


def _rtail(series, threshold):
    """Returns the right-most (i.e. towards the end) indices of the given
    series such that the sum of their values is less than threshold. Useful
    for creating 'other' categories from nominal summaries or '>x' categories
    from ordinal summaries.
    
    Parameters:
        series - a pandas Series
        threshold - the desired cumulative weight for the tail
    """
    # Create a "reversed" cumulative sum, from back to front
    rev_cumsum = series - series.cumsum() + series.sum()
    return series.index[rev_cumsum < threshold]


def _ltail(series, threshold):
    """Returns the left-most (i.e. towards the beginning) indices of the given
    series such that the sum of their values is less than threshold. Useful
    for creating '<x' categories from ordinal summaries.
    
    Parameters:
        series - a pandas Series
        threshold - the desired cumulative weight for the tail
    """
    return series.index[series.cumsum() < threshold]


def s_trim_l(series, threshold, newcat=None):
    """Trim the left end of a series. Find the left-most elements whose sum
    is less than 'threshold', remove them, and then add their total under a
    new index called 'newcat' at the beginning of the series. If one or fewer
    elements woudld be trimmed, don't do anything."""
    tidx = _ltail(series, threshold) # Tail InDeXes
    # Only do anything if there's more than one index to combine
    if len(tidx) > 1:
        # Default new category name is "<= (last index removed)"
        if newcat is None:
            newcat = "<= {}".format(tidx[-1])
        cap = pandas.Series(data=[series.loc[tidx].sum()], index=[newcat])
        return pandas.concat([cap, series.drop(tidx)])
    else:
        return series


def s_trim_r(series, threshold, newcat=None):
    """Trim the right end of a series. Find the right-most elements whose sum
    is less than 'threshold', remove them, and then add their total under a
    new index called 'newcat' at the end of the series. If one or fewer
    elements woudld be trimmed, don't do anything."""
    tidx = _rtail(series, threshold) # Tail InDeXes
    # Only do anything if there's more than one index to combine
    if len(tidx) > 1:
        # Default new category name is ">= (first index removed)"
        if newcat is None:
            newcat = ">= {}".format(tidx[0])
        cap = pandas.Series(data=[series.loc[tidx].sum()], index=[newcat])
        return pandas.concat([series.drop(tidx), cap])
    else:
        return series


def df_trim_i_l(df, threshold, newcat=None):
    """Trim the top rows of a dataframe. Find the top-most row indices, for
    which the sum of all elements in those rows is less than 'threshold',
    remove them, and add their column totals as a new row called 'newcat' at
    the top of the dataframe. If one or fewer rows would be triemmed,
    don't do anything."""
    tidx = _ltail(df.sum(axis=1), threshold) # Tail InDeXes
    # Only do anything if there's more than one index to combine
    if len(tidx) > 1:
        if newcat is None:
            newcat = "<= {}".format(tidx[-1])
        # Column sums for all the dropped rows
        to_add = df.loc[tidx,:].sum(axis=0)
        to_add.name = newcat
        return pandas.concat([pandas.DataFrame(to_add).T, df.drop(tidx, axis=0)])
    else:
        return df


def df_trim_i_r(df, threshold, newcat=None):
    """Trim the bottom rows of a dataframe. Find the bottom-most row indices,
    for which the sum of all elements in those rows is less than 'threshold',
    remove them, and add their column totals as a new row called 'newcat' at
    the bottom of the dataframe. If one or fewer rows would be triemmed,
    don't do anything."""
    tidx = _rtail(df.sum(axis=1), threshold) # Tail InDeXes
    # Only do anything if there's more than one index to combine
    if len(tidx) > 1:
        if newcat is None:
            newcat = ">= {}".format(tidx[0])
        # Column sums for all the dropped rows
        to_add = df.loc[tidx,:].sum(axis=0)
        to_add.name = newcat
        return df.drop(tidx, axis=0).append(to_add)
    else:
        return df


def df_trim_c_l(df, threshold, newcat=None):
    return df_trim_i_l(df.T, threshold, newcat=newcat).T


def df_trim_c_r(df, threshold, newcat=None):
    return df_trim_i_r(df.T, threshold, newcat=newcat).T

