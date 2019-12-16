# -*- coding: utf-8 -*-
"""
predictor.py

Contains the base class for the prediction framework based on posterior
predictive samples of game outcomes
"""

from . import analysis
from .. import util

import pandas

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
            raise Exception("Invalid value for 'mode': '{}'".format(mode))
        # A list of all the unique index values (chain, draw)
        if self._echo:
            print("Assembling indices...")
        self._indices = list(set((c,d) for c,d,_ in self._modeldf.index))
        # Create lists to hold data frames, tables, and matrices
        self._df_save = None
        self._table_save = None
        self._matrix_save = None
        # Dict to map statistic names to stat keys
        self._name2stat = {}
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
        return list(self._name2stat.keys())
    
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
        try:
            precompute = stat.precompute
        except AttributeError:
            pass
        # If stat has a type_ attribute, override was was passed in
        try:
            type_ = stat.type_
        except AttributeError:
            pass
        # If stat has a sort attribute, override what was passed in
        try:
            sort = stat.sort
        except AttributeError:
            pass
        # If there was no name passed in, get it from the stat
        if name is None:
            try:
                # Directly from a name attribute, or ...
                name = stat.name
            except AttributeError:
                # ... just from the __name__
                name = stat.__name__
        # Is this name already used?
        if (name in self.stats) and (self._name2stat.get(name) != stat):
            raise Exception("Name {} already used by stat".format(name))
        elif (name not in self.stats):
            # Register the stat's name
            self._name2stat[name] = stat
        # Compute if we need to
        if stat not in self._stat_values:
            # Echo if needed
            if self._echo:
                print("Calculating {}...".format(name))
            # Register the stat's type
            self._stat_types[stat] = type_
            if type_ == 'ordinal':
                self._stat_sort[stat] = sort
            # Compute and store the stat's values
            if precompute == 'df':
                self._stat_values[stat] = [stat(df) for df in self._dataframes]
            elif precompute == 'table':
                self._stat_values[stat] = [stat(t) for t in self._tables]
            elif precompute == 'matrix':
                self._stat_values[stat] = [stat(m) for m in self._matrices]
            else:
                raise Exception("Invalid value for 'precompute': '{}'".format(precompute))
    
    # Methods to summarize statistics
    
    def summary(self, names=None):
        """Compute summaries of desired statistic(s).
        Parameters:
            names - either a string or a list of strings, name(s) of stats
        Returns:
            If stat is a string, a single pandas Series containing the summary.
            If stat is a list of strings, a dict with stat names as keys and
            summaries as values.
        """
        # By default, summarize all stats
        if names is None:
            names = self.stats
        # Create summary
        summ = None
        if type(names) == str: # For single stat
            if self._stat_types[self._name2stat[names]] == 'numeric':
                summ = self._summary_numeric(names)
            elif self._stat_types[self._name2stat[names]] == 'ordinal':
                summ = self._summary_ordinal(names)
            elif self._stat_types[self._name2stat[names]] == 'nominal':
                summ = self._summary_nominal(names)
        elif type(names) == list: # For list of stats
            summ = {}
            for n in names:
                if self._stat_types[self._name2stat[n]] == 'numeric':
                    summ[n] = self._summary_numeric(self._name2stat[n])
                elif self._stat_types[self._name2stat[n]] == 'ordinal':
                    summ[n] = self._summary_ordinal(self._name2stat[n])
                elif self._stat_types[self._name2stat[n]] == 'nominal':
                    summ[n] = self._summary_nominal(self._name2stat[n])
        return summ
    
    def _summary_numeric(self, stat):
        """Summarize a statistic whose type is 'numeric'. Doesn't validate.
        Parameters:
            stat - a stat key to the self._stat_values dict"""
        return pandas.Series(self._stat_values[stat])\
            .describe(percentiles=[.025,.25,.5,.75,.975])
    
    def _summary_nominal(self, stat):
        """Summarize a statistic whose type is 'nominal'. Doesn't validate.
        Parameters:
            stat - a stat key to the self._stat_values dict"""
        df = pandas.DataFrame({stat:self._stat_values[stat], 'count':1})
        c = df.groupby(stat).count()['count']
        # Sort by count descending, normalize
        return c.sort_values(ascending=False) / len(df)
    
    def _summary_ordinal(self, stat):
        """Summarize a statistic whose type is 'ordinal'. Doesn't validate.
        Parameters:
            stat - a stat key to the self._stat_values dict"""
        df = pandas.DataFrame({stat:self._stat_values[stat], 'count':1})
        c = df.groupby(stat).count()['count']
        # Sort by category ascending, normalize
        return c[sorted(c.index, key=self._stat_sort[stat])] / len(df)
    
    # Methods to plot statistics
    
    def plot(self, names=None, nrows=1, ncols=1, figsize=None):
        """Make plots of desired statistic(s).
        Parameters:
            names - either a string or a list of strings, name(s) of stats
            nrows, ncols - the number of rows and columns to use to arrange
                the axes in each figure.
            figsize - a tuple for the figure size.
        Returns:
            A generator which yields figures one at a time.
        """
        # By default, summarize all stats
        if names is None:
            names = self.stats
        # For each stat, plot it on an axis
        if type(names) == str:
            names = list(names)
        # Make a generator for statistics to zip with the axes
        namegen = (n for n in names)
        # Get the generator for figures we will draw on
        figs = util.make_axes(len(names), nrows, ncols, figsize, "Statistic Plots")
        for fig in figs:
            for ax, n in zip(fig.axes, namegen):
                if self._stat_types[self._name2stat[n]] == 'numeric':
                    self._plot_numeric(n, ax)
                elif self._stat_types[self._name2stat[n]] in ['ordinal','nominal']:
                    self._plot_categorical(n, ax)
            yield fig
    
    def _plot_numeric(self, name, ax):
        """Plot a stat whose type is 'numeric'. Doesn't validate.
        Parameters:
            name - a stat name, a key in self._name2stat"""
        ax.hist(self._stat_values[self._name2stat[name]], density=True)
        ax.set_title(name)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(name)
        return ax
    
    def _plot_categorical(self, name, ax):
        """Plot a stat whose type is 'nominal' or 'ordinal'. Doesn't validate.
        Parameters:
            name - a stat name, a key in self._name2stat"""
        s = self.summary(name)
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
        ax.set_title(name)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(name)
        # If the combined length of the labels is too long, rotate the labels
        if sum(len(str(x)) for x in s.index) > maxtextlen:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        return ax
    
    def to_dataframe(self):
        """Convert the statistics calculated by this predictor to a dataframe.
        Returns: a df with one row per posterior sample, with columns 'chain' 
        and 'draw' indicating the sample, and one column per statistic."""
        df = pandas.DataFrame({n:self._stat_values[s] for n,s in self._name2stat.items()})
        df['chain'] = [c for c,d in self._indices]
        df['draw'] = [d for c,d in self._indices]
        return df.set_index(['chain','draw']).sort_index()

        