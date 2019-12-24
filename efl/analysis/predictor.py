# -*- coding: utf-8 -*-
"""
predictor.py

Contains the base class for the prediction framework based on posterior
predictive samples of game outcomes
"""

from .. import util

import pandas
import warnings
import itertools

###########################################################
### CLASS FOR MAKING PREDICTIONS ##########################
###########################################################

class EFLPredictor(object):
    """EFLPredictor - framework for computing summaries of posterior predictive
    samples from EFL models."""
    
    def __init__(self, model, mode=None):
        """Parameters:
            model - subclass of EFLModel
            mode - what mode this predictor will operate in. can be one of the
                following values: (1) 'past', use predicted results for
                observed games and nothing for unobserved games. (2) 'future',
                use nothing for observed games and predicted results for
                unobserved games. (3) 'full', use predicted results for both
                observed and unobserved games. (4) 'mix', use observed results
                for observed games, and predicted results for unobserved games.
                If None, will be either 'mix', 'past', or 'future' depending
                on whether model has fitgameids and/or predictgameids.
        """
        # Determine default mode
        if mode is None:
            if len(model.fitgameids) > 0 and len(model.predictgameids) > 0:
                mode = "mix"
            elif len(model.fitgameids) > 0:
                mode = "past"
            elif len(model.predictgameids) > 0:
                mode = "future"
            else:
                raise Exception("model has no games?")
        # Determine which results to use on which subsets of games
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
        self._indices = list(set((c,d) for c,d,_ in self._modeldf.index))
        # Create list to hold one data frame per sampling iteration
        self._samples_save = None
        # Dict to hold computed statistics
        self._stat_values = {}
        # Dict to hold references to precomputes for stats
        self._stat_precompute = {}
        # Dict to hold the data types of computed statistics
        self._stat_type = {}
        # Dict to hold the sort keys for ordinal types
        self._stat_sort = {}
        # Dict to map statistic names to stat keys
        self._name2stat = {}
        # Dict of dicts to map stat group names to their substats and substat
        # names
        self._groups = {}
    
    # Property to lazy compute and retrieve the data frames for each iteration
    
    @property
    def _samples(self):
        if self._samples_save is None:
            # For each posterior draw, take the predicted results and append
            # the observed results, if any
            self._samples_save = [self._modeldf.loc[i,:].append(self._gamesdf) for i in self._indices]
        return self._samples_save
    
    # Property that lists the stats that have been added
    
    @property
    def names(self):
        return list(set(self._name2stat.keys()) | set(self._groups.keys()))
    
    # Methods to compute and add stats to this object
    
    def add_stat(self, stat, name=None, pool=None):
        """Add a statistic to this Predictor.
        Parameters:
            stat - a callable that will be used to compute the statistic
            name - The name to use when saving and referencing this stat
            pool - An optional multiprocessing pool to split computation of
                values between multiple cores/processes
        Notes:
            If stat is callable, it will be considered a single stat. If it has
            attributes "precompute", "type_", or "sort", those will determine
            how this stat is handled. If stat is not callable, then it is
            assumed to be dict-like and considered a stat group. The name
            passed to this function is always used, if any. If stat has a
            "name" attribute, it will serve as the default name if none is
            passed to this function. Otherwise, its __name__ attribute is the
            default.
        """
        # If there was no name passed in, get it from the stat
        name = self._determine_name(stat, name)
        # Is this name already used?
        if (name in self.names) and \
                (self._name2stat.get(name) != stat) and \
                (self._groups.get(name) != stat):
            raise Exception("Name '{}' already used by another stat".format(name))
        # Is this a grouped stat or a single stat? Use callability to decide
        if callable(stat):
            # Compute if we need to, along with any prerequisites
            self._compute_stat(stat, pool)
            # Register the name. Note: okay if name already exists b/c to get here
            # it must already refer to this stat
            self._name2stat[name] = stat
            # Check if this stat has a type, if not, warn
            if self._stat_type.get(stat) is None:
                warnings.warn("Stat '{}' has no type. It will not be able "
                              "to be plotted or summarized, but it may be "
                              "returned raw by to_dataframe()".format(name))
        else:
            # Assume this is a substat, dict-like
            # Compute all of the substats
            for subname,substat in stat.items():
                # Compute the stat and any prerequisites
                self._compute_stat(substat, pool)
                # Warn if stats do not have types
                if self._stat_type.get(substat) is None:
                    warnings.warn("Stat '{}/{}' has no type. It will not be able "
                                  "to be plotted or summarized, but it may be "
                                  "returned raw by to_dataframe()".format(name, subname))
            # Store statgroup itself as map
            self._groups[name] = stat
    
    @staticmethod
    def _determine_name(x, name):
        """Determine the appropriate name to use for a stat x, following the
        standard hierarchy of priotity."""
        if name is None:
            if getattr(x, 'name', None) is not None:
                name = x.name
            else:
                name = x.__name__
        return name
    
    def _compute_stat(self, stat, pool=None):
        """Computes a statistic and registers its type, precompute(s), and
        sort key. Recursively calls itself for precomputes. Does not register
        any names."""
        # Only proceed if this stat hasn't been computed
        if stat in self._stat_values:
            return
        # check for a precompute attribute
        precompute = getattr(stat, 'precompute', None)
        try:
            precompute = tuple(precompute)
        except TypeError:
            if precompute is not None:
                precompute = (precompute,)
        # Check for a type_ attribute
        type_ = getattr(stat, 'type_', None)
        # Check for a sort attribute
        sort = getattr(stat, 'sort', None)
        # Now compute this statistic
        if precompute is None:
            if pool is None:
                self._stat_values[stat] = [stat(df) for df in self._samples]
            else:
                self._stat_values[stat] = pool.map(stat, self._samples)
        else:
            # Compute all necessary prerequisites
            for pc in precompute:
                self._compute_stat(pc, pool)
            # Zip together precomputed values and pass as positional arguments
            pcvals = (self._stat_values[pc] for pc in precompute)
            if pool is None:
                self._stat_values[stat] = [stat(*args) for args in zip(*pcvals)]
            else:
                self._stat_values[stat] = pool.starmap(stat, zip(*pcvals))
        # Save the precompute info
        if precompute is not None:
            self._stat_precompute[stat] = precompute
        # Save the type info
        if type_ is not None:
            self._stat_type[stat] = type_
        # Save the sort info
        if (type_ == 'ordinal') and (sort is not None):
            self._stat_sort[stat] = sort
    
    # Method to return samples as a dataframe
    
    def to_dataframe(self, names=None):
        """Convert the statistics calculated by this predictor to a dataframe.
        Returns: a df with one row per posterior sample, with columns 'chain' 
        and 'draw' indicating the sample, and one column per statistic.
        Parameters:
            stats - a string or list of strings containing names of stats to
                put in dataframe. Defaults to all.
        """
        # Convert stat parameter to list format, defaulting to all
        if names is None:
            names = self.names
        elif type(names) == str:
            names = [names]
        # Check that all names are present in this Predictor
        missing = list(set(names) - set(self.names))
        if len(missing) > 0:
            raise Exception("Stats ({}) are not present.".format(", ".join(missing)))
        # Create DataFrame and set index
        df = pandas.DataFrame({('chain',''):[c for c,d in self._indices],
                               ('draw',''):[d for c,d in self._indices]})
        for n in names:
            if n in self._name2stat: # If a solo stat, add to DataFrame
                df[(n,'')] = self._stat_values[self._name2stat[n]]
            elif n in self._groups: # If a group, add each substat to DF
                for sn,ss in self._groups[n].items():
                    df[(n,sn)] = self._stat_values[ss]
        return df.set_index(['chain','draw']).sort_index()
    
    # Methods to remove stats and clean the object
    
    def remove_stat(self, name):
        raise NotImplementedError()
    
    def clean(self):
        raise NotImplementedError()
    
#    # Methods to summarize statistics
    
    def summary(self, names=None):
        """Compute summaries of desired statistic(s).
        Parameters:
            names - either a string or a list of strings, name(s) of stats
        Returns:
            A dict with stat names as keys and summaries as values.
        """
        # By default, summarize all stats
        if names is None:
            names = self.names
        # If only one provided, make it a list
        if type(names) == str:
            names = [names]
        # Create summary
        summ = {}
        for n in names:
            if n in self._name2stat:
                summ[n] = self._summary_single(self._name2stat[n], n)
            elif n in self._groups:
                # For groups, summarize each pair of substats together
                # Combine into a dict
                grpsum = {}
                for sn1,sn2 in itertools.combinations(self._groups[n], 2):
                    ss1 = self._groups[n][sn1]
                    ss2 = self._groups[n][sn2]
                    grpsum[(sn1, sn2)] = self._summary_pair(ss1, sn1, ss2, sn2)
                summ[n] = grpsum
        return summ
    
    def _summary_single(self, stat, name):
        """Summarize a single stat, by name, and return its summary."""
        if self._stat_type[stat] == 'numeric':
            return self._summary_numeric(stat, name)
        else:
            return self._summary_categorical(stat, name)
    
    def _summary_numeric(self, stat, name):
        """Summarize a statistic whose type is 'numeric'. Doesn't validate.
        Parameters:
            stat - a stat key to the self._stat_values dict
            name - name by which to refer to the stat"""
        return pandas.Series(self._stat_values[stat], name=name)\
            .describe(percentiles=[.025,.25,.5,.75,.975])
    
    def _summary_categorical(self, stat, name):
        """Summarize a statistic whose type is 'nominal' or 'ordinal'. Checks
        type to determine how to sort the summary.
        Parameters:
            stat - a stat key to the self._stat_values dict
            name - a name by which to refer to the stat"""
        df = pandas.DataFrame({name:self._stat_values[stat], 'cnt':1})
        c = df.groupby(name)['cnt'].count()
        # Sort appropriately, return
        if self._stat_type[stat] == 'nominal':
            return c.sort_values(ascending=False) / len(df)
        elif self._stat_type[stat] == 'ordinal':
            return c[sorted(c.index, key=self._stat_sort.get(stat))] / len(df)
        else:
            raise Exception("Stat '{}' type neither nominal nor ordinal.".format(name))
    
    def _summary_pair(self, xstat, xname, ystat, yname):
        """Summarize a pair of stats, by name, and return the summary."""
        if self._stat_type[xstat] == self._stat_type[ystat] == 'numeric':
            return self._summary_num_num(xstat, xname, ystat, yname)
        elif self._stat_type[xstat] == 'numeric':
            return self._summary_num_cat(xstat, xname, ystat, yname)
        elif self._stat_type[ystat] == 'numeric':
            return self._summary_cat_num(xstat, xname, ystat, yname)
        else:
            return self._summary_cat_cat(xstat, xname, ystat, yname)
    
    def _summary_num_num(self, xstat, xname, ystat, yname):
        """Summarize a pair with both numeric stats."""
        df = pandas.DataFrame({xname:self._stat_values[xstat],
                               yname:self._stat_valeus[ystat]})
        # TODO: add measures of correlation, etc.
        return df.describe(percentiles=[.025,.25,.5,.75,.975])
        
    def _summary_num_cat(self, xstat, xname, ystat, yname):
        """Summarize a pair with one numeric stat and one categorical stat.
        Return the summary so that the numeric stat is on the "x-axis" and
        the categorical stat is on the "y-axis".
        Parameters:
            xstat, xname - the numeric stat in the pair
            ystat, yname - the categorical stat in the pair
        """
        df = pandas.DataFrame({xname:self._stat_values[xstat],
                               yname:self._stat_values[ystat]})
        summ = df.groupby(ystat).describe(percentiles=[.025,.25,.5,.75,.975])
        csumm = self._summary_cat(ystat, yname)
        del summ[(xname,'count')]
        df.insert(0, 'frequency', csumm)
        # Sort in whatever order the categorical summary sorts in (it handles
        # nominal vs ordinal)
        return summ.loc[csumm.index,:]
    
    def _summary_cat_num(self, xstat, xname, ystat, yname):
        """Summarize a pair with one numeric stat and one categorical stat.
        Return the summary so that the categorical stat is on the "x-axis" and
        the numeric stat is on the "y-axis".
        Parameters:
            xstat, xname - the categorical stat in the pair
            ystat, yname - the numeric stat in the pair
        """
        return self._summary_num_cat(ystat, yname, xstat, xname).T
    
    def _summary_cat_cat(self, xstat, xname, ystat, yname, totals=True):
        """Summarize a pair with both categorical stats."""
        df = pandas.DataFrame({xname:self._stat_values[xstat],
                               yname:self._stat_values[ystat],
                               'cnt':1})
        summ = df.pivot_table(values='cnt', index=yname, columns=xname,
                              aggfunc='count', fill_value=0) / len(df)
        xsumm = self._summary_categorical(xstat, xname)
        ysumm = self._summary_categorical(ystat, yname)
        summ = summ.loc[ysumm.index, xsumm.index]
        if totals:
            summ.loc['Total',:] = xsumm
            summ.loc[:,'Total'] = ysumm
            summ.loc['Total','Total'] = 1.0
        return summ

    # Methods to plot statistics
    
    def _statgen_maker(self, names2gen):
        """Take a supplied list of names and return tuples giving instructions
        to plot() about what to plot on an axis."""
        for n in names2gen:
            if n in self._name2stat:
                yield (self._name2stat[n], n)
            else:
                for sn1,sn2 in itertools.combinations(self._groups[n], 2):
                    yield (n, self._groups[n][sn1], sn1, 
                           self._groups[n][sn2], sn2)
    
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
            names = self.names
        # Make names a list for convenience
        if type(names) == str:
            names = list(names)
        # How many plots are we going to have to make?
        numplots = 0
        for n in names:
            if n in self._name2stat:
                numplots += 1
            elif n in self._groups:
                num_ss = len(self._groups[n])
                numplots += num_ss * (num_ss - 1) // 2
        # Make a generator of the stats we need to plot
        statgen = self._statgen_maker(names)
        # Get the generator for figures we will draw on
        figs = util.make_axes(numplots, nrows, ncols, figsize, "Statistic Plots")
        for fig in figs:
            # Bug fix: the order (fig.axes, statgen) is important as fig.axes
            # is shorter and elements in statgen were getting "wasted" during
            # zipping. https://docs.python.org/3/library/functions.html#zip
            for ax, args in zip(fig.axes, statgen):
                if len(args) == 2: # We want to plot a single stat
                    self._plot_single(*args, ax)
                else: # We have a pair to plot
                    self._plot_pair(*args, ax)
            yield fig
    
    def _plot_single(self, stat, name, ax):
        """Plot a single stat on the given axis, and return the axis."""
        if self._stat_type[stat] == 'numeric':
            return self._plot_numeric(stat, name, ax)
        else:
            return self._plot_categorical(stat, name, ax)
    
    def _plot_numeric(self, stat, name, ax):
        """Plot a stat whose type is 'numeric'. Doesn't validate.
        Parameters:
            stat - a stat key to the self._stat_values dict
            name - a name by which to refer to the stat
            ax - matplotlib Axes object on which to draw"""
        ax.hist(self._stat_values[stat], density=True, bins="fd")
        util.draw_densplot(ax, self._stat_values[stat])
        ax.set_title(name)
        ax.set_ylabel("Frequency")
        return ax
    
    def _plot_categorical(self, stat, name, ax):
        """Plot a stat whose type is 'nominal' or 'ordinal'. Doesn't validate.
        Parameters:
            stat - a stat key to the self._stat_values dict
            name - a name by which to refer to the stat
            ax - matplotlib Axes object on which to draw
        """
        # Total text length of labels before rotating
        maxtextlen = 40
        # Get the appropriate summary
        s = self._summary_categorical(stat, name)
        # Make a bar plot out of the summary
        bars = ax.bar(x=range(1,len(s)+1), height=s, tick_label=s.index)
        # Print relative frequencies over bars
        if 3*len(s) > maxtextlen:
            rotation=90
        else:
            rotation=0
        for bar in bars:
            height = bar.get_height()
            xc = bar.get_x() + bar.get_width()/2
            ax.text(x=xc, y=height, s="{:.0%}".format(height),
                    ha='center', va='bottom', rotation=rotation)
        # Set titles and labels
        ax.set_title(name)
        ax.set_ylabel("Frequency")
        # If the combined length of the labels is too long, rotate the labels
        if sum(len(str(x)) for x in s.index) > maxtextlen:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        return ax
    
    def _plot_pair(self, title, xstat, xname, ystat, yname, ax):
        """Plot a pair of stats on an axis and return the axis."""
        if self._stat_type[xstat] == self._stat_type[ystat] == 'numeric':
            return self._plot_num_num(title, xstat, xname, ystat, yname, ax)
        elif self._stat_type[xstat] == 'numeric':
            return self._plot_num_cat(title, xstat, xname, ystat, yname, ax)
        elif self._stat_type[ystat] == 'numeric':
            return self._plot_cat_num(title, xstat, xname, ystat, yname, ax)
        else:
            return self._plot_cat_cat(title, xstat, xname, ystat, yname, ax)
    
    def _plot_num_num(self, title, xstat, xname, ystat, yname, ax):
        raise NotImplementedError()
        
    def _plot_num_cat(self, title, xstat, xname, ystat, yname, ax):
        raise NotImplementedError()
    
    def _plot_cat_num(self, title, xstat, xname, ystat, yname, ax):
        raise NotImplementedError()
    
    def _plot_cat_cat(self, title, xstat, xname, ystat, yname, ax):
        """Plot two categorical variables as an imagemap.
        Parameters:
            xstat, xname - the stat in the pair that will end up in columns
            ystat, yname - the stat in the pair that will end up in rows
        """
        data = self._summary_cat_cat(xstat, xname, ystat, yname, totals=False)
        util.heatmap(data, ax)
        ax.set_title(title)
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        return ax
