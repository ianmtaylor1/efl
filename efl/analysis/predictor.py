# -*- coding: utf-8 -*-
"""
predictor.py

Contains the base class for the prediction framework based on posterior
predictive samples of game outcomes
"""

from . import analysis

import pandas
import matplotlib.pyplot as plt
import itertools

###########################################################
### CLASS FOR MAKING PREDICTIONS ##########################
###########################################################

class EFLPredictor(object):
    """EFLPredictor - framework for computing summaries of posterior predictive
    samples from EFL models."""
    
    def __init__(self, model, mode="mix", echo=True):
        """Parameters:
            model - subclass of EFLModel
            mode - what mode this predictor will operate in. can be one of the
                following values: (1) 'past', use predicted results for
                observed games and nothing for unobserved games. (2) 'future',
                use nothing for observed games and predicted results for
                unobserved games. (3) 'full', use predicted results for both
                observed and unobserved games. (4) 'mix', use observed results
                for observed games, and predicted results for unobserved games.
            echo - Should computation updates be printed? Improves user sanity
        """
        self._echo = echo
        # Determine which results to use on which subsets of games
        if self._echo:
            print("Extracting predictions...")
        if mode == "past":
            self._modeldf = model.predict("fit")
            self._gamesdf = None
        elif mode == "future":
            self._modeldf = model.predict("predict")
            self._gamesdf = None
        elif mode == "full":
            self._modeldf = model.predict("all")
            self._gamesdf = None
        elif mode == "mix":
            self._modeldf = model.predict("predict")
            self._gamesdf = model.gamedata.loc[model.fitgameids,:]
        else:
            raise Exception("Invalid value for parameter 'mode'")
        # A list of all the unique index values (chain, draw)
        if self._echo:
            print("Assembling indices...")
        self._indices = list(set((c,d) for c,d,_ in self._modeldf.index))
        # Create lists to hold data frames, tables, and matrices
        self._df_save = None
        self._table_save = None
        self._matrix_save = None
        # Dict to hold computed statistics
        self._stat_values = {}
        # Dict to hold the data types of computed statistics
        self._stat_types = {}
        # Dict to hold the sort keys for ordinal types
        self._stat_sort = {}
    
    # Properties to lazy compute and retrieve the data frames, tables, or
    # matrices needed by stat functions
    
    @property
    def _dataframes(self):
        if self._df_save is None:
            if self._echo:
                print("Precomputing game lists...")
            # For each posterior draw, take the predicted results and append
            # the observed results, if any
            self._df_save = [self._modeldf.loc[i,:].append(self._gamesdf) for i in self._indices]
            if self._echo:
                print("Done precomputing lists...")
        return self._df_save
    
    @property
    def _tables(self):
        if self._table_save is None:
            if self._echo:
                print("Precomputing league tables...")
            self._table_save = [analysis.make_table(df) for df in self._dataframes]
            if self._echo:
                print("Done precomputing tables...")
        return self._table_save
    
    @property
    def _matrices(self):
        if self._matrix_save is None:
            if self._echo:
                print("Precomputing directed win graphs...")
            self._matrix_save = [analysis.make_matrix(df) for df in self._dataframes]
            if self._echo:
                print("Done precomputing graphs...")
        return self._matrix_save
    
    # Property that lists the stats that have been added
    @property
    def stats(self):
        return list(self._stat_values.keys())
    
    def add_stat(self, stat, name=None, precompute="df", type_="nominal", sort=None):
        """Add a statistic to this Predictor.
        Parameters:
            stat - a callable that will be used to compute the statistic
            name - The name to use when saving and referencing this stat
            precompute - what needs to be precomputed and passed to this stat.
                ('df', 'table' or 'matrix')
            type_ - the return type of this stat ('ordinal', 'nominal', or
                'numeric')
            sort - if type_ is ordinal, a key function to sort the values
        Notes:
            If stat has attributes "precompute", "type_", or "sort", those will
            override what is passed to this function. Name behaves differently.
            The name passed to this function is always used, if any. If stat 
            has a "name" attribute, it will serve as the default name if none
            is passed to this function. Otherwise, its __name__ attribute is
            the default.
        """
        # If stat has a precompute attribute, override what was passed in
        if 'precompute' in stat.__dict__:
            precompute = stat.precompute
        # If stat has a type_ attribute, override was was passed in
        if 'type_' in stat.__dict__:
            type_ = stat.type_
        # If stat has a sort attribute, override what was passed in
        if 'sort' in stat.__dict__:
            sort = stat.sort
        # If there was no name passed in, get it from the stat
        if name is None:
            if 'name' in stat.__dict__: # Directly from a name attribute, or
                name = stat.name
            else:  # Just from the __name__
                name = stat.__name__
        # Is this name already used?
        if name in self.stats:
            raise Exception("Name {} already used by stat".format(name))
        # Echo if needed
        if self._echo:
            print("Calculating {}...".format(name))
        # Register the stat's type
        self._stat_types[name] = type_
        if type_ == 'ordinal':
            self._stat_sort[name] = sort
        # Compute and store the stat's values
        if precompute == 'df':
            self._stat_values[name] = [stat(df) for df in self._dataframes]
        elif precompute == 'table':
            self._stat_values[name] = [stat(t) for t in self._tables]
        elif precompute == 'matrix':
            self._stat_values[name] = [stat(m) for m in self._matrices]
        else:
            raise Exception("'{}' is invalid value for 'precompute'".format(precompute))
    
    # Methods to summarize statistics
    
    def summary(self, stat=None):
        """Compute summaries of desired statistic(s).
        Parameters:
            stat - either a string or a list of strings, name(s) of stats
        Returns:
            If stat is a string, a single pandas Series containing the summary.
            If stat is a list of strings, a dict with stat names as keys and
            summaries as values.
        """
        # By default, summarize all stats
        if stat is None:
            stat = self.stats
        # Create summary
        summ = None
        if type(stat) == str: # For single stat
            if self._stat_types[stat] == 'numeric':
                summ = self._summary_numeric(stat)
            elif self._stat_types[stat] == 'ordinal':
                summ = self._summary_ordinal(stat)
            elif self._stat_types[stat] == 'nominal':
                summ = self._summary_nominal(stat)
        elif type(stat) == list: # For list of stats
            summ = {}
            for s in stat:
                if self._stat_types[s] == 'numeric':
                    summ[s] = self._summary_numeric(s)
                elif self._stat_types[s] == 'ordinal':
                    summ[s] = self._summary_ordinal(s)
                elif self._stat_types[s] == 'nominal':
                    summ[s] = self._summary_nominal(s)
        return summ
    
    def _summary_numeric(self, stat):
        """Summarize a statistic whose type is 'numeric'. Doesn't validate."""
        return pandas.Series(self._stat_values[stat])\
            .describe(percentiles=[.025,.25,.5,.75,.975])
    
    def _summary_nominal(self, stat):
        """Summarize a statistic whose type is 'nominal'. Doesn't validate."""
        df = pandas.DataFrame({stat:self._stat_values[stat], 'count':1})
        c = df.groupby(stat).count()['count']
        # Sort by count descending, normalize
        return c.sort_values(ascending=False) / len(df)
    
    def _summary_ordinal(self, stat):
        """Summarize a statistic whose type is 'ordinal'. Doesn't validate."""
        df = pandas.DataFrame({stat:self._stat_values[stat], 'count':1})
        c = df.groupby(stat).count()['count']
        # Sort by category ascending, normalize
        return c[sorted(c.index, key=self._stat_sort[stat])] / len(df)
    
    # Methods to plot statistics
    
    def plot(self, stat=None, page=(1,1)):
        """Make plots of desired statistic(s).
        Parameters:
            stat - either a string or a list of strings, name(s) of stats
            page - tuple describing how many plots to put in one figure, 
                arranged in (rows, columns)
        Returns:
            A list of matplotlib.pyplot figures for the plotted statistics.
        """
        # By default, summarize all stats
        if stat is None:
            stat = self.stats
        # For each stat, plot it on an axis
        if type(stat) == str:
            stat = list(stat)
        figs, axes = _make_axes(len(stat), page, "Statistic Plots")
        for ax,s in zip(axes,stat):
            if self._stat_types[s] == 'numeric':
                self._plot_numeric(s, ax)
            elif self._stat_types[s] in ['ordinal','nominal']:
                self._plot_categorical(s, ax)
        return figs
    
    def _plot_numeric(self, stat, ax):
        """Plot a stat whose type is 'numeric'. Doesn't validate."""
        ax.hist(self._stat_values[stat], density=True)
        ax.set_title(stat)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(stat)
        return ax
    
    def _plot_categorical(self, stat, ax):
        """Plot a stat whose type is 'nominal' or 'ordinal'. Doesn't validate."""
        s = self.summary(stat)
        bars = ax.bar(x=range(1,len(s)+1), height=s, tick_label=s.index)
        maxtextlen = 40
        # Print relative frequencies over bars
        digits = 3
        if (digits+2)*len(s) > maxtextlen:
            rotation=90
        else:
            rotation=0
        for bar in bars:
            height = bar.get_height()
            xc = bar.get_x() + bar.get_width()/2
            ax.text(x=xc, y=height, s=str(round(height,digits)),
                    ha='center', va='bottom', rotation=rotation)
        # Set titles and labels
        ax.set_title(stat)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(stat)
        # If the combined length of the labels is too long, rotate the labels
        if sum(len(str(x)) for x in s.index) > maxtextlen:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        return ax
    
    def to_dataframe(self):
        """Convert the statistics calculated by this predictor to a dataframe.
        Returns: a df with one row per posterior sample, with columns 'chain' 
        and 'draw' indicating the sample, and one column per statistic."""
        df = pandas.DataFrame(self._stat_values)
        df['chain'] = [c for c,d in self._indices]
        df['draw'] = [d for c,d in self._indices]
        return df.set_index(['chain','draw']).sort_index()


###############################################################################
## UTILITY FUNCTIONS ##########################################################
###############################################################################

 
def _make_axes(num_plots, page, main_title):
    """Create figures and plots to be used by the plotting functions of
    EFLPredictor.
    Parameters:
        num_plots - number of plots that will be plotted
        page - a 2-tuple for the number of (rows, columns) that the axes will
            be arranged in per page.
        main_title - the title of each figure created. If None, no title will
            be added to the plot. Else, a title in the format
            'main_title #/#' will be added indicating the number of the plot
            in the sequence
    Returns: (figs, axes), a list of figures and a list of axes. figs contains
        one figure per actual figure created. len(axes) is always num_plots,
        and the list axes contains all the axes to be plotted on.
    """
    numfigs = (num_plots // (page[0]*page[1])) + ((num_plots % (page[0]*page[1])) > 0)
    subplots = [plt.subplots(nrows=page[0], ncols=page[1]) for i in range(numfigs)]
    figs = [x[0] for x in subplots]
    if (page[0]*page[1] == 1):
        axes = [x[1] for x in subplots]
    else:
        axes = list(itertools.chain.from_iterable(x[1].flatten() for x in subplots))
    for ax in axes[num_plots:]:
        ax.axis('off')
    axes = axes[:num_plots]
    for i,f in enumerate(figs):
        if main_title is not None:
            f.suptitle("{} {}/{}".format(main_title, i+1, numfigs))
        f.subplots_adjust(hspace=0.5, wspace=0.25)
    return figs, axes