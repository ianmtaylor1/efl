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


def _find_level(Z, quantile, tol=.001, maxiter=50):
    """Find thresholds for contours that enclose the given quantile as
    a highest-density set. Works as a binary search, narrowing in on the
    right value. Used for contour2d"""
    total = Z.sum()
    # Initial values
    low = 0
    high = Z.max()
    current = (low + high) / 2
    lowp = Z[Z >= low].sum() / total
    highp = Z[Z >= high].sum() / total
    currentp = Z[Z >= current].sum() / total
    i = 0 # Iteration counter
    while (i < maxiter) and (lowp - highp >= tol):
        i += 1
        if currentp > quantile: # Need to move the threshold up (enclose less)
            low = current
            lowp = currentp
            current = (current + high) / 2
        elif currentp < quantile: # Need to move the thresh down (enclose more)
            high = current
            highp = currentp
            current = (current + low) / 2
        currentp = Z[Z >= current].sum() / total
    # We are either within tolerance or have exceeded iteration cap
    return current


def contour2d(ax, data, res=110, credsets=[0.5, 0.95], **kwargs):
    """Draw 2d contours based on the KDE of the given data on the axes
    provided.
    Parameters:
        ax - the axes to draw on
        data - a numpy array of dimension (2,n)
        res - the "resolution" in each dimension of the grid on which the KDE
            will be computed
        credsets - draw contours enclosing these proportions of the kernel
            densty estimate (credible set levels)
        **kwargs - passed to ax.contour()
    """
    # Make the density estimator
    density = kde.gaussian_kde(data, bw_method="scott")
    # Create the grid on which to evaluate the density
    xlow = min(data[0]) - 0.05*(max(data[0]) - min(data[0]))
    xhigh = max(data[0]) + 0.05*(max(data[0]) - min(data[0]))
    ylow = min(data[1]) - 0.05*(max(data[1]) - min(data[1]))
    yhigh = max(data[1]) + 0.05*(max(data[1]) - min(data[1]))
    xgrid = numpy.arange(xlow, xhigh, (xhigh - xlow)/res)
    ygrid = numpy.arange(ylow, yhigh, (yhigh - ylow)/res)
    # Evaluate the density
    evalx = numpy.tile(xgrid, len(ygrid))
    evaly = numpy.repeat(ygrid, len(xgrid))
    flatZ = density(numpy.array([evalx, evaly]))
    Z = numpy.reshape(flatZ, (len(ygrid), len(xgrid)))
    # Compute and draw the contours
    credsets = sorted(credsets, key=lambda x: -x) # Reverse sort
    levels = [_find_level(Z, p) for p in credsets]
    ctrs = ax.contour(xgrid, ygrid, Z, levels=levels, **kwargs)
    ax.clabel(ctrs, fmt={lv:'{:.0%}'.format(t) for lv,t in zip(levels,credsets)})
    # Return the contours, I guess?
    return ctrs


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
        valfmt - A format string, function, or Formatter instance to use
            formatting both the annotated values (if any) and the colorbar
            ticks (if any)
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
    if type(valfmt) == str:
        formatter = matplotlib.ticker.StrMethodFormatter(valfmt)
    elif callable(valfmt):
        formatter = matplotlib.ticker.FuncFormatter(valfmt)
    else:
        formatter = valfmt
    
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
    if total_xlab_len > 2*maxtextlen:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va="center",
                 rotation_mode="anchor")
    elif total_xlab_len > maxtextlen:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va="top",
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
        threshold = (im.norm(data_array.min()) + im.norm(data_array.max()))/2
        # Annotate each cell
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                textcolor = textcolors[int(im.norm(data_array[i, j]) > threshold)]
                im.axes.text(j, i, formatter(data_array[i, j], None),
                             color=textcolor, horizontalalignment="center", 
                             verticalalignment="center")
    
    # Return the axes we just drew on
    return im


def barplot(data, ax, xlabels=None, annotate=True, maxtextlen=40,
            valfmt='{x:.0%}', **kwargs):
    """Draw a bar plot of the given data on the given axes.
    Parameters:
        data - a pandas.Series containing the data to plot
        ax - the matplotlib axes object on which to plot
        xlabels - the labels to draw on the x axis. If None, the index of the
            data series will be used.
        annotate - if True, write percentages above bars
        maxtextlen - if total length of text of xlabels is greater than this
            number, rotate labels 90 degrees
        valfmt - A format string to use formatting both the annotated values
            (if any) and y axis ticks.
    """
    # Default labels
    if xlabels is None:
        xlabels = [str(x) for x in data.index]
    # Draw bars
    bars = ax.bar(x=range(len(data)), height=data, tick_label=xlabels)
    # Format values
    formatter = matplotlib.ticker.StrMethodFormatter(valfmt)
    ax.yaxis.set_major_formatter(formatter)
    # Annotations
    if annotate:
        # Print relative frequencies over bars
        # Rotate if necessary
        if 3*len(data) > maxtextlen:
            rotation=90
        else:
            rotation=0
        for bar in bars:
            height = bar.get_height()
            xc = bar.get_x() + bar.get_width()/2
            ax.text(x=xc, y=height, s=formatter(height),
                    ha='center', va='bottom', rotation=rotation)
        # Extend the y limit to accomodate bars
        ax.set_ylim(bottom=0.0, top=max(data)*1.2)
    # If the combined length of the labels is too long, rotate the labels
    if sum(len(str(x)) for x in xlabels) > maxtextlen:
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
    # Return the bars
    return bars


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
            # Unicode less-than-or-equal
            newcat = "\u2264 {}".format(tidx[-1])
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
            # Unicode greater-than-or-equal
            newcat = "\u2265 {}".format(tidx[0])
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
            # Unicode less-than-or-equal
            newcat = "\u2264 {}".format(tidx[-1])
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
            # Unicode greater-than-or-equal
            newcat = "\u2265 {}".format(tidx[0])
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


###########################################################
### PRIOR MAKING HELPER FUNCTIONS #########################
###########################################################


def shuffle_rename(df, cols, newnames=None, replace=False, inplace=True):
    """For each row of the given dataframe, shuffle the values in the
    indicated columns. If newnames is not None, then also rename those columns.
    If inplace is False, copy the dataframe first. Returnes the manipulated
    DataFrame.
    Parameters:
        df - pandas DataFrame to operate on
        cols - list of column names to shuffle
        newnames - list of names to rename cols to
        replace - if True, sample with replacement. If False, shuffle.
        inplace - if False, copy df before operating and return the new copy.
    """
    # Do we need to copy?
    if not inplace:
        df = df.copy()
    # Do the shuffling
    rg = numpy.random.default_rng()
    df.loc[:,cols] = numpy.apply_along_axis(
            rg.choice, 1, df.loc[:,cols],
            replace=replace, size=len(cols))
    # Rename if necessary
    if newnames is not None:
        colmap = {old:new for old,new in zip(cols, newnames)}
        df.rename(columns=colmap, inplace=True) # Always in place
    # If we made a copy, return it
    if not inplace:
        return df


def mean_var(df, cols=None, center=True, meanregress=1, varspread=1, ridge=0):
    """Calculate a mean array and covariance matrix for the given columns
    of the given dataframe.
    Parameters:
        df - pandas DataFrame to use
        cols - columns of interest. If None, use all columns
        center - if True, subtract overall mean to center means at zero.
        meanregress - Shrink means towards overall mean by this factor.
            mean = avg(mean) + meanregress * (mean - avg(mean))
        varspread - multiply covariance matrix by this factor
        ridge - add this value to all entries on the diagonal of the
            covariance matrix (after any spread)
    """
    # Default: all columns
    if cols is None:
        cols = df.columns
    # Basic mean and var
    mean = numpy.array(df.loc[:,cols].mean())
    var = numpy.cov(df.loc[:,cols].T)
    # Transform mean as required
    mean = mean.mean() + meanregress * (mean - mean.mean())
    if center:
        mean = mean - mean.mean()
    # Transform variance as required
    var = var * varspread + ridge * numpy.identity(len(cols))
    # Return the results
    return mean,var

