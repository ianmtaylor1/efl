#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base.py 

Module contains classes that models inherit from, implementing useful
common features.
"""

from . import cache
from .. import util

import numpy
import math
import pandas
import itertools
import re
import matplotlib


###############################################################################
## BASE MODEL #################################################################
###############################################################################


class Model(object):
    """Base class for EFL models. Mostly handles wrapping of the StanFit
    object, since inheritance is hard.
    
    This class provides/creates:
        1. An __init__ method that builds and fits the model
        2. Instance attribute: _model 
        3. Instance attribute: stanfit
        4. Instance attribute: _modeldata (equal to whatever was passed to
                __init__). Useful for _stan_inits function
        5. Methods: summary, stansummary - wraps self.stanfit.summary() and
                self.stanfit.stansummary(), and replaces uninformative
                paramater names with more informative ones from _stan2efl
        6. Method: predict - return posterior predictive samples of game
                scores and outcomes.
        7. Method: to_dataframe - wraps self.stanfit.to_dataframe(), and
                replaces parameter names same as summary().
        8. Readable attributes fitgameids and predictgameids which are lists
                of all game id's used to fit, and predicted by this
                model (respectively).
        9. A readable attribute gamedata, which is the data for games included
                in this model, in dataframe form. (As it would be returned by
                EFLGames.to_dataframe)
        10. A readable attribute parameters, which is a list of all efl (human-
                readable) parameters available in the model.
        11. Plotting methods: traceplot, densplot, boxplot, acfplot
    
    Subclasses of EFLModel should:
        1. Provide a method _predict() for predicting a single game outcome.
    
    Subclasses of EFLModel may:
        1. Implement _stan_inits() method for generating initial values 
                from chain_id. The method can be passed to the init argument
                of the base constructor.
    """
    
    def __init__(self, modelfile, modeldata, fitgameids, predictgameids,
                 gamedata, efl2stan, pargroups={}, init='random',
                 chains=4, samples=10000, warmup=2000, thin=4, n_jobs=1,
                 **kwargs):
        """Initialize the base properties of this model.
        Parameters:
            modelfile - the filename of the stan model to use. Used in the
                cache module to fetch precompiled models.
            modeldata - Data that is passed to sampling(data=modeldata). Is
                also stored as self._modeldata for use in other methods.
            fitgameids, predictgameids - lists of game ids which were used to
                fit the model, or which are predicted by the model.
            gamedata - The data for games included in this model, in dataframe
                form. (As it would be returned by EFLGames.to_dataframe)
            efl2stan - a dict with keys that are human-readable model
                parameters, and values that are the corresponding Stan
                parameters. Should contain every parameter available to be
                returned by "summary" or "to_dataframe".
            pargroups - a dict with keys that are parameter group names, and
                values that are lists of the human-readable parameter names in
                that group. (i.e. values are lists containing some of the keys
                of efl2stan.)
            init - argument passed to pystan sampling() method. Default is
                'random'. A method of the model class can also be created and
                passed in as this argument. It will be called for each chain,
                per pystan docs.
            chains, warmup, thin, n_jobs - same as pystan options. Passed to 
                sampling()
            samples - number of desired posterior samples, total from all
                chains, after warmup and thinning. Used to calculate iter, and
                then pass iter to sampling()
            **kwargs - any additional keyword arguments to pass to sampling()
        """
        # Get model from compiled model cache
        self._model = cache.get_model(modelfile)
        # Store the data that was passed as an instance attribute
        self._modeldata = modeldata
        # Save the fit and predict game id's, as well as game data
        self.fitgameids = fitgameids
        self.predictgameids = predictgameids
        self.gamedata = gamedata
        # Create parameter mappings
        self._efl2stan = efl2stan
        self._stan2efl = dict(reversed(i) for i in self._efl2stan.items())
        self._pargroups = pargroups
        # Calculate the total iterations needed
        iter_ = warmup + math.ceil(samples * thin / chains)
        # Fit the model
        self.stanfit = self._model.sampling(
            data=self._modeldata, init=init, chains=chains, iter=iter_,
            warmup=warmup, thin=thin, n_jobs=n_jobs, **kwargs)
    
    # Stubs for methods that should be implemented by subclasses
    
    def _predict(self, gameid, **kwargs):
        """Predicts the outcome of a single game. Returns a pandas.DataFrame
        with (at least) the columns:
            chain - chain number of the sample
            draw - draw number within the chain
            homegoals - home goals, if predicted by model (NA otherwise)
            awaygoals - away goals, if predicted by model (NA otherwise)
            result - match result ('H','A','D')
        """
        raise NotImplementedError(
                "_predict not implemented in {}".format(type(self))
                )
    
    # Methods and properties provided by this base class
    
    def _translate_pars(self, pars=None):
        """Method to translate human-readable parameters or parameter groups
        into a list of stan parameters to pass to the methods of the stan
        fit.
        Parameters:
            pars - string or list of strings. If string, can be the name of
            a parameter, parameter group, or the word 'all'. If a list, must
            be a list of parameter names.
        Returns:
            list of parameter names corresponding to 'pars'.
        """
        # If pars is none, replace it with the word "all"
        if pars is None:
            pars = 'all'
        # Standardize the input into a list of human-readable parameter names
        if type(pars) == str:
            if pars in self.pargroups:
                # This is a parameter group
                eflparlist = self._pargroups[pars]
            elif pars == 'all':
                # This is the keyword "all". This check is second to allow
                # models to override the 'all' keyword in _pargroups
                eflparlist = self.parameters
            elif pars == 'other':
                # This is the keyword 'other'. This keyword references all
                # parameters that are not part of any defined group. 
                # Check is done here to allow models to override in _pargroups
                tmppars = set(self.parameters)
                for g in self.pargroups:
                    tmppars -= set(self._pargroups[g])
                eflparlist = list(tmppars)
            else:
                # This is a parameter by itself
                eflparlist = [pars]
        else:
            # Assume this is a list or iterable
            eflparlist = pars
        # Now reference the mapping dictionary to get stanfit parameters
        return [self._efl2stan[p] for p in eflparlist]         
    
    @property
    def parameters(self):
        return list(self._efl2stan.keys())
    
    @property
    def pargroups(self):
        return list(self._pargroups.keys())
    
    def stansummary(self, pars=None, **kwargs):
        """A wrapper around the stansummary method on the included stanfit
        object. It will convert parameter names as defined in _stan2efl, and
        by default will only include those parameters which are keys in 
        that dict."""
        # Fill default pars and map to stan names
        stspars = self._translate_pars(pars)
        # Run stansummary for the underlying fit
        sts = self.stanfit.stansummary(pars=stspars, **kwargs)
        # Translate summary to useful parameter names
        addlength = max(len(self._stan2efl[p]) for p in stspars) \
                    - max(len(p) for p in stspars) + 1
        for (stanpar, eflpar) in self._stan2efl.items():
            spaces = addlength - (len(eflpar) - len(stanpar))
            if spaces >= 0: # Need to net insert spaces
                old = r'^{} '.format(re.escape(stanpar))
                new = '{}{} '.format(eflpar, " "*spaces)
            else: # Need to net remove spaces
                old = r'^{}{} '.format(re.escape(stanpar), " "*abs(spaces))
                new = eflpar+' '
            sts = re.sub(old, new, sts, flags=re.MULTILINE)
        # Also add spaces at the start of the header row
        if addlength > 0:
            sts = sts.replace(" mean ", "{} mean ".format(" "*addlength))
        elif addlength < 0:
            sts = sts.replace("{} mean ".format(" "*abs(addlength)), " mean ")
        return sts
    
    def summary(self, pars=None, by_chain=False):
        """Returns a summary of the posterior distribution as a pandas
        DataFrame with rows for parameters and columns for statistics."""
        # Fill default pars and map to stan names
        stspars = self._translate_pars(pars)
        # Run stanfit's summary method for the underlying fit
        sts = self.stanfit.summary(pars=stspars)
        # Transform into form we want
        summ = None
        if by_chain:
            nchains = sts['c_summary'].shape[2]
            summlist = [None] * nchains
            # Create a dataframe for each chain
            for c in range(nchains):
                summlist[c] = pandas.DataFrame(
                        sts['c_summary'][:,:,c],
                        columns = sts['c_summary_colnames'])
                summlist[c]['chain'] = c
                summlist[c]['parameter'] = [self._stan2efl[p] for p in sts['c_summary_rownames']]
            # Combine them and set parameter,chain as index
            summ = pandas.concat(summlist).set_index(['parameter','chain'])
        else:
            summ = pandas.DataFrame(
                    sts['summary'],
                    columns = sts['summary_colnames'],
                    index = [self._stan2efl[p] for p in sts['summary_rownames']]
                    )
        return summ
    
    def to_dataframe(self, pars=None, **kwargs):
        """A wrapper around the to_dataframe method on the included stanfit
        object. It will convert column par names as defined in _stan2efl, and
        by default will only include those parameters which are keys in 
        that dict."""
        # Fill default pars and map to stan names
        stspars = self._translate_pars(pars)
        # Run to_dataframe for the underlying fit
        df = self.stanfit.to_dataframe(pars=stspars, **kwargs)
        # Translate the column names to useful parameter names
        df.columns = [self._stan2efl.get(c, c) for c in df.columns]
        return df
    
    def diagnostics(self, **kwargs):
        """Return per-sample diagnostics for the underlying stanfit. Wraps
        the stanfit's to_dataframe method with no pars and diagnostics=True.
        All other keyword arguments (e.g. permuted, inc_warmup) are passed to
        to_dataframe."""
        return self.stanfit.to_dataframe(pars=[], diagnostics=True, **kwargs)
    
    def predict(self, gameids="all", **kwargs):
        """Predicts the outcome of a group of games. 
        Parameters:
            gameids - either a keyword ('fit', 'predict', 'all') or an
                iterable of game ids to predict
            **kwargs - any extra keyword arguments for specific subclasses to
                implement
        Returns these games in a pandas.DataFrame with (chain, draw, gameid) as
        a multi-index, and the following columns (in order):
            date - date the game took place
            hometeam - unique short name of home team
            awayteam - unique short name of away team
            homegoals - home goals, if available (NA otherwise)
            awaygoals - away goals, if available (NA otherwise)
            result - match result ('H','A','D') if available (NA otherwise)
        """
        # Sort out keyword arguments
        if (gameids == "all") or (gameids is None):
            gameids = itertools.chain(self.fitgameids, self.predictgameids)
        elif gameids == "fit":
            gameids = self.fitgameids
        elif gameids == "predict":
            gameids = self.predictgameids
        # Predict each game, assign gameid, and concatenate
        df = pandas.concat(
                (self._predict(g, **kwargs).assign(gameid=g) for g in gameids),
                ignore_index=True)\
                .merge(self.gamedata[['date','hometeam','awayteam']], 
                       left_on='gameid', right_index=True, validate="m:1", 
                       how='left')\
                .set_index(['chain','draw','gameid'])\
                .sort_index()
        return df[['date','hometeam','awayteam','homegoals','awaygoals','result']]
    
    def autocorr(self, pars=None, lags=[1,5,10,20]):
        """Computes autocorrelations of posterior samples from this model.
        Used for assessing MCMC samples.
        Parameters:
            pars - parameters for which to compute autocorrelation
            lags - iterable of lags to use
        Returns:
            A pandas.DataFrame with one column per parameter containing
            computed autocorrelations. The index is the supplied lags
        """
        # Get the data for autocorr
        samples = self.to_dataframe(pars, diagnostics=False, permuted=False)
        eflpars = [c for c in samples.columns if c in self.parameters]
        # Create blank df and compute autocorr
        ac = pandas.DataFrame(numpy.NaN, columns=eflpars, index=lags)
        for p in eflpars:
            # Can this be done faster with numpy.correlate?
            for n in lags:
                ac.loc[n,p] = samples[p].autocorr(n)
        return ac
    
    def corr(self, pars=None):
        """Computes the correlation for all supplied parameters from the
        posterior samples.
        Parameters:
            pars - parameters for which to compute correlations
        Returns:
            A pandas.DataFrame with parameters as columns and row indices.
            Values are correlation between row and column parameters.
        """
        samples = self.to_dataframe(pars, diagnostics=False, permuted=False)
        eflpars = [c for c in samples.columns if c in self.parameters]
        return samples[eflpars].corr()
    
    def traceplot(self, pars=None, combine_chains=False, 
                  nrows=2, ncols=2, figsize=None):
        """Create a traceplot of the samples from this model.
        Parameters:
            pars - list of parameters to make plots for. Default all.
            combine_chains - True: combine chains and plot one series per
                graph. False: plot each chain as individual series on each
                graph.
            nrows, ncols - the number of rows and columns to use to arrange
                the axes in each figure.
        Returns: a generator which yields figures one at a time        
        """
        # Get the data for plotting
        samples = self.to_dataframe(pars, diagnostics=False, permuted=False)
        numpars = sum(1 for c in samples.columns if c in self.parameters)
        eflpars = (c for c in samples.columns if c in self.parameters)
        # Create the figures and axes for plotting
        figs = util.make_axes(numpars, nrows, ncols, figsize, "Traceplot")
        for fig in figs:
            # Draw traceplots on all the axes objects
            # Since eflpars is a generator, this will pick up where it left
            # off in the parameters with each new figure.
            for ax,p in zip(fig.axes, eflpars):
                ax.set_title(p)
                if combine_chains:
                    ax.plot(range(len(samples[p])), samples[p], linewidth=1)
                else:
                    nchains = samples['chain'].max() + 1
                    for c in samples['chain'].unique():
                        chain = samples.loc[samples['chain'] == c, p]
                        ax.plot(range(len(chain)), chain, linewidth=1/nchains)
            # Yield this figure once done drawing
            yield fig
    
    def densplot(self, pars=None, combine_chains=False,
                 nrows=2, ncols=2, figsize=None):
        """Create a density plot of the samples from this model.
        Parameters:
            pars - list of parameters to make plots for. Default all.
            combine_chains - True: combine chains and plot one series per
                graph. False: plot each chain as individual series on each
                graph.
            nrows, ncols - the number of rows and columns to use to arrange
                the axes in each figure.
        Returns: a generator which yields figures one at a time
        """
        # Get the data for plotting
        samples = self.to_dataframe(pars, diagnostics=False, permuted=False)
        numpars = sum(1 for c in samples.columns if c in self.parameters)
        eflpars = (c for c in samples.columns if c in self.parameters)
        # Create the figures and axes for plotting
        figs = util.make_axes(numpars, nrows, ncols, figsize, "Density Plot")
        for fig in figs:
            # Draw density estimates on all of the axes
            # Since eflpars is a generator, this will pick up where it left
            # off in the parameters with each new figure.
            for ax,p in zip(fig.axes, eflpars):
                ax.set_title(p)
                if combine_chains:
                    util.draw_densplot(ax, samples[p])
                else:
                    for c in samples['chain'].unique():
                        chain = samples.loc[samples['chain'] == c, p]
                        util.draw_densplot(ax, chain)
            # Return the list of figures
            yield fig
    
    def boxplot(self, pars=None, vert=True, figsize=None):
        """Create side-by-side boxplots of the posterior samples from this
        model.
        Parameters:
            pars - list of parameters to make plots for. Default all.
            vert - should the boxplots be vertical?
        Returns: a matplotlib.pyplot figure
        """
        # Get the data for plotting
        samples = self.to_dataframe(pars, diagnostics=False, permuted=False)
        eflpars = [c for c in samples.columns if c in self.parameters]
        # Create the figures and axes for plotting
        # Here, we know that this will only create exactly one figure with
        # exactly one Axes object.
        fig = next(util.make_axes(1, 1, 1, figsize, None))
        ax = fig.axes[0]
        # Draw the boxplots on these axes
        ax.set_title("Parameter Boxplot")
        ax.boxplot(samples[eflpars].T, labels=eflpars, 
                   showmeans=True, showcaps=False, showfliers=False,
                   whis=[2.5, 97.5], vert=vert)
        if vert:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        # Return the figure (no list, since boxplot always is on one figure)
        return fig
    
    def acfplot(self, pars=None, maxlag=20, nrows=2, ncols=2, figsize=None):
        """Create an acf plot for the posterior samples of this model.
        Parameters:
            pars - the parameters to plot autocorrelation for. Default all.
            maxlag - the maximum lag to compute and plot autocorelation for.
            nrows, ncols - the number of rows and columns to use to arrange
                the axes in each figure.
        """
        # Find autocorrelations. Let autocorr do parameter parsing, and just
        # look at what columns it returns
        ac = self.autocorr(pars, range(maxlag+1))
        numpars = len(ac.columns)
        eflpars = (c for c in ac.columns)
        # Create the figures and axes for plotting
        figs = util.make_axes(numpars, nrows, ncols, figsize, 
                              "Autocorrelation Plot")
        for fig in figs:
            # Draw the autocorrelation plots
            # Since eflpars is a generator, this will pick up where it left
            # off in the parameters with each new figure.
            for ax,p in zip(fig.axes, eflpars):
                ax.set_title(p)
                ax.bar(ac.index, ac[p])
            yield fig
    
    def corrplot(self, pars=None, fixscale=True, annotate=False, figsize=None):
        """Plot a correlation matrix of the supplied parameters, as calculated
        by corr()
        Parameters:
            pars - parameters for which to compute correlations
            fixscale - If True, the color scale is fixed to (-1, 1). If false,
                it is rescaled to the observed correlation range.
            annotate - If True, annotate the plot with the calculated
                correlations.
        Returns:
            A matplotlib figure showing the correlation plot.
        """
        # Calculate correlation matrix
        cmat = self.corr(pars)
        data = cmat.to_numpy()
        rlab = list(cmat.index)
        clab = list(cmat.columns)
        # Create figure and axis
        fig = next(util.make_axes(1, 1, 1, figsize, None))
        ax = fig.axes[0]
        # Plot matrix on axis
        # Fill the diagonal with an out-of-range value, then adjust cmap and
        # norm accordingly.
        numpy.fill_diagonal(data, 0)
        lim = 1.0 if fixscale else min(abs(data).max()*1.1, 1.0)
        numpy.fill_diagonal(data, 1.01)
        cmap = matplotlib.pyplot.get_cmap('RdBu')
        cmap.set_over('black')
        norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
        # Format values to remove leading zeros and the out-of-range value
        def fmat(val, pos):
            return '{:.2f}'.format(val).replace("0.",".").replace("1.01","")
        # Draw the heatmap
        util.heatmap(data, ax, row_labels=rlab, col_labels=clab,
                     cbarlabel="Correlation Coefficient", norm=norm,
                     valfmt=fmat, textcolors=['black','black'], cmap=cmap,
                     annotate=annotate)
        return fig
    
    def scatterplot(self, pars=None, credsets=[0.25, 0.5, 0.75, 0.95], 
                    nrows=2, ncols=2, figsize=None):
        """Create scatterplots of posterior samples for each pair of the
        supplied pars, along with HPD credible sets.
        Parameters:
            pars - the parameters to make scatterplots for. Default all.
            credsets - draw contours representing the HPD credible sets for
                these confidence levels.
            nrows, ncols - the number of rows and columns to use to arrange
                the axes in each figure.
        """
        # Get the data for plotting
        samples = self.to_dataframe(pars, diagnostics=False, permuted=False)
        eflpars = [c for c in samples.columns if c in self.parameters]
        # Find the number of pairs we will need to plot
        numplots = len(eflpars) * (len(eflpars) - 1) // 2
        # An iterator over the pairs
        pairs = itertools.combinations(eflpars, 2)
        # Create figures
        figs = util.make_axes(numplots, nrows, ncols, figsize,
                              "Scatterplot")
        for fig in figs:
            # Draw scatterplot on each axes in each figure
            for ax, (p1, p2) in zip(fig.axes, pairs):
                ax.scatter(samples[p1], samples[p2], marker=".", alpha=0.25)
                util.contour2d(ax, samples[[p1,p2]].to_numpy().T, colors="k",
                               credsets=credsets)
                ax.set_title("{} vs {}".format(p2, p1))
                ax.set_xlabel(p1)
                ax.set_ylabel(p2)
            yield fig


###############################################################################
## HELPER MODELS ##############################################################
###############################################################################


class GoalModel(Model):
    """Base class for models that predict home/away goals of games."""
    
    def __init__(self, eflgames, extramodeldata={}, **kwargs):
        """Parameters:
            eflgames - an EFLGames instance
            extramodeldata - a dict which will be merged with the standard
                data format created by this class. (Usually priors.)
        """
        # Create game id's
        fitgameids     = [g.id for g in eflgames.fit]
        predictgameids = [g.id for g in eflgames.predict]
        # Create the prediction quantity names
        self._predictqtys = {}
        for i,gid in enumerate(fitgameids):
            self._predictqtys[gid] = ('homegoals_pred[{}]'.format(i+1),
                                      'awaygoals_pred[{}]'.format(i+1))
        for i,gid in enumerate(predictgameids):
            self._predictqtys[gid] = ('homegoals_new_pred[{}]'.format(i+1),
                                      'awaygoals_new_pred[{}]'.format(i+1))
        # Get the standard model data format
        stddata = self._get_model_data(eflgames)
        # Call base model, passing along the fit and predict game ids
        super().__init__(modeldata      = {**stddata, **extramodeldata},
                         fitgameids     = fitgameids,
                         predictgameids = predictgameids,
                         gamedata       = eflgames.to_dataframe(fit=True, predict=True),
                         **kwargs)
    
    @staticmethod
    def _get_model_data(eflgames):
        """Take an EFLGames instance and transform it into a dict appropriate
        for the standard input of Stan models that predict game goals. This
        format guarantees:
            * The observed and new games are indexed in the order they appear
              in eflgames.fit and eflgames.predict, respectively.
            * The teams are indexed in the order they appear in eflgames.teams
        """
        # Number of games and teams
        nGames = len(eflgames.fit)
        nGames_new = len(eflgames.predict)
        nTeams = len(eflgames.teams)
        # Team indexes for each game
        teamidxmap = {t.id:(i+1) for i,t in enumerate(eflgames.teams)}
        hometeamidx = numpy.array(
                [teamidxmap[g.hometeamid] for g in eflgames.fit], 
                dtype=numpy.int_)
        awayteamidx = numpy.array(
                [teamidxmap[g.awayteamid] for g in eflgames.fit], 
                dtype=numpy.int_)
        hometeamidx_new = numpy.array(
                [teamidxmap[g.hometeamid] for g in eflgames.predict], 
                dtype=numpy.int_)
        awayteamidx_new = numpy.array(
                [teamidxmap[g.awayteamid] for g in eflgames.predict], 
                dtype=numpy.int_)
        # Game goals
        homegoals = numpy.array(
                [g.result.homegoals for g in eflgames.fit],
                dtype=numpy.int_)
        awaygoals = numpy.array(
                [g.result.awaygoals for g in eflgames.fit],
                dtype=numpy.int_)
        # Return the model data
        return {'nGames':nGames, 'nGames_new':nGames_new, 'nTeams':nTeams,
                'hometeamidx':hometeamidx, 'awayteamidx':awayteamidx,
                'homegoals':homegoals, 'awaygoals':awaygoals,
                'hometeamidx_new':hometeamidx_new, 
                'awayteamidx_new':awayteamidx_new}
    
    def _predict(self, gameid):
        """Predict the result for the game 'gameid' from this model's fitted 
        data."""
        # Find the quantity we need to look at
        hg, ag = self._predictqtys[gameid]
        # Pull that quantity
        samples = self.stanfit.to_dataframe(pars=[hg, ag], permuted=False,
                                            diagnostics=False)
        # Map to a result
        samples['homegoals'] = samples[hg].map(int)
        samples['awaygoals'] = samples[ag].map(int)
        samples['result'] = 'D'
        samples.loc[(samples['homegoals']>samples['awaygoals']),'result'] = 'H'
        samples.loc[(samples['homegoals']<samples['awaygoals']),'result'] = 'A'
        # Drop quantity and return
        return samples.drop([hg, ag], axis=1)


class ResultModel(Model):
    """Base class for models that predict just game results."""
    
    def __init__(self, eflgames, extramodeldata={}, **kwargs):
        """Parameters:
            eflgames - an EFLGames instance
            extramodeldata - a dict which will be merged with the standard
                data format created by this class. (Usually priors.)
        """
        # Create game id's
        fitgameids     = [g.id for g in eflgames.fit]
        predictgameids = [g.id for g in eflgames.predict]
        # Create the prediction quantity names
        self._predictqtys = {}
        for i,gid in enumerate(fitgameids):
            self._predictqtys[gid] = 'result_pred[{}]'.format(i+1)
        for i,gid in enumerate(predictgameids):
            self._predictqtys[gid] = 'result_new_pred[{}]'.format(i+1)
        # Get the standard model data format
        stddata = self._get_model_data(eflgames)
        # Call base model, passing along the fit and predict game ids
        super().__init__(modeldata      = {**stddata, **extramodeldata},
                         fitgameids     = fitgameids, 
                         predictgameids = predictgameids,
                         gamedata       = eflgames.to_dataframe(fit=True, predict=True),
                         **kwargs)
    
    @staticmethod
    def _get_model_data(eflgames):
        """Take an EFLGames instance and transform it into a dict appropriate
        for the standard input of Stan models that predict game results. This
        format guarantees:
            * The observed and new games are indexed in the order they appear
              in eflgames.fit and eflgames.predict, respectively.
            * The teams are indexed in the order they appear in eflgames.teams
        """
        # Number of games and teams
        nGames = len(eflgames.fit)
        nGames_new = len(eflgames.predict)
        nTeams = len(eflgames.teams)
        # Team indexes for each game
        teamidxmap = {t.id:(i+1) for i,t in enumerate(eflgames.teams)}
        hometeamidx = numpy.array(
                [teamidxmap[g.hometeamid] for g in eflgames.fit], 
                dtype=numpy.int_)
        awayteamidx = numpy.array(
                [teamidxmap[g.awayteamid] for g in eflgames.fit], 
                dtype=numpy.int_)
        hometeamidx_new = numpy.array(
                [teamidxmap[g.hometeamid] for g in eflgames.predict], 
                dtype=numpy.int_)
        awayteamidx_new = numpy.array(
                [teamidxmap[g.awayteamid] for g in eflgames.predict], 
                dtype=numpy.int_)
        # Game results
        def result_int(g): # Function to return the appropriate integers
            if g.result.homegoals > g.result.awaygoals:
                return 3 # Home win
            elif g.result.homegoals < g.result.awaygoals:
                return 1 # Home loss
            else:
                return 2 # Draw
        result = numpy.array(
                [result_int(g) for g in eflgames.fit],
                dtype=numpy.int_)
        # Return the model data
        return {'nGames':nGames, 'nGames_new':nGames_new, 'nTeams':nTeams,
                'hometeamidx':hometeamidx, 'awayteamidx':awayteamidx,
                'result':result,
                'hometeamidx_new':hometeamidx_new, 
                'awayteamidx_new':awayteamidx_new}
    
    def _predict(self, gameid):
        """Predict the result for the game 'gameid' from this model's fitted 
        data."""
        # Find the quantity we need to look at
        qtyname = self._predictqtys[gameid]
        # Pull that quantity
        samples = self.stanfit.to_dataframe(pars=[qtyname], permuted=False,
                                            diagnostics=False)
        # Map to a result
        samples['result'] = samples[qtyname].apply(
                lambda x: ['A','D','H'][int(x)-1])
        samples['homegoals'] = numpy.NaN
        samples['awaygoals'] = numpy.NaN
        # Drop quantity and return
        return samples.drop(qtyname, axis=1)

