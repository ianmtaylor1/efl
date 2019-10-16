#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base.py 

Module contains classes that models inherit from, implementing useful
common features.
"""

from . import cache

import numpy
import pandas
import itertools
import re
import matplotlib.pyplot as plt
from scipy.stats import kde

###############################################################################
## BASE MODEL #################################################################
###############################################################################


class EFLModel(object):
    """Base class for EFL models. Mostly handles wrapping of the StanFit
    object, since inheritance is hard.
    
    This class provides/creates:
        1. An __init__ method that builds and fits the model
        2. Instance attribute: _model 
        3. Instance attribute: stanfit
        4. Instance attribute: _modeldata (equal to whatever was passed to
                __init__). Useful for _stan_inits function
        5. Method: summary - wraps self.stanfit.stansummary(), and replaces 
                uninformative paramater names with more informative ones
                from _stan2efl
        6. Method: to_dataframe - wraps self.stanfit.to_dataframe(), and
                replaces parameter names same as summary().
        7. Readable attributes fitgameids and predictgameids which are lists
                of all game id's used to fit, and predicted by this
                model (respectively).
        8. A readable attribute parameters, which is a list of all efl (human-
                readable) parameters available in the model.
    
    Subclasses of EFLModel should:
        1. Implement _stan_inits() method for generating initial values 
                from chain_id
        2. Provide a method _predict() for predicting a single game outcome.
    """
    
    def __init__(self, modelfile, modeldata, fitgameids, predictgameids,
                 efl2stan,
                 chains=4, iter=12000, warmup=2000, thin=4, n_jobs=1,
                 **kwargs):
        """Initialize the base properties of this model.
        Parameters:
            modelfile - the filename of the stan model to use. Used in the
                cache module to fetch precompiled models.
            modeldata - Data that is passed to sampling(data=modeldata). Is
                also stored as self._modeldata for use in other methods.
            fitgameids, predictgameids - lists of game ids which were used to
                fit the model, or which are predicted by the model.
            efl2stan - a dict with keys that are human-readable model
                parameters, and values that are the corresponding Stan
                parameters. Should contain every parameter available to be
                returned by "summary" or "to_dataframe".
            chains, iter, warmup, thin, n_jobs - same as pystan options.
                Passed to sampling()
            **kwargs - any additional keyword arguments to pass to sampling()
        """
        # Get model from compiled model cache
        self._model = cache.get_model(modelfile)
        # Store the data that was passed as an instance attribute
        self._modeldata = modeldata
        # Save the fit and predict game id's
        self.fitgameids = fitgameids
        self.predictgameids = predictgameids
        # Create parameter mappings
        self._efl2stan = efl2stan
        self._stan2efl = dict(reversed(i) for i in self._efl2stan.items())
        # Fit the model
        self.stanfit = self._model.sampling(
            data=self._modeldata, init=self._stan_inits,
            chains=chains, iter=iter, warmup=warmup, thin=thin, n_jobs=n_jobs,
            **kwargs)
    
    # Stubs for methods that should be implemented by subclasses
    
    def _stan_inits(self, chain_id=None):
        """Produce initial values for MCMC. Should return a dict with keys
        equal to Stan model parameters, and values equal to their initial
        values."""
        raise NotImplementedError(
                "_stan_inits not implemented in {}".format(type(self))
                )
    
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
    
    @property
    def parameters(self):
        return list(self._efl2stan.keys())
    
    def stansummary(self, pars=None, **kwargs):
        """A wrapper around the stansummary method on the included stanfit
        object. It will convert parameter names as defined in _stan2efl, and
        by default will only include those parameters which are keys in 
        that dict."""
        # Fill default pars and map to stan names
        if pars is None:
            pars = self._efl2stan.keys()
        stspars = [self._efl2stan[p] for p in pars]
        # Run stansummary for the underlying fit
        sts = self.stanfit.stansummary(pars=stspars, **kwargs)
        # Translate summary to useful parameter names
        addlength = max(len(p) for p in self._stan2efl.values()) \
                    - max(len(p) for p in self._stan2efl.keys()) + 1
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
        if pars is None:
            pars = self._efl2stan.keys()
        stspars = [self._efl2stan[p] for p in pars]
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
        if pars is None:
            pars = self._efl2stan.keys()
        stspars = [self._efl2stan[p] for p in pars]
        # Run to_dataframe for the underlying fit
        df = self.stanfit.to_dataframe(pars=stspars, **kwargs)
        # Translate the column names to useful parameter names
        df.columns = [self._stan2efl.get(c, c) for c in df.columns]
        return df
    
    def predict(self, gameids="all", **kwargs):
        """Predicts the outcome of a group of games. 
        Parameters:
            gameids - either a keyword ('fit', 'predict', 'all') or an
                iterable of game ids to predict
            **kwargs - any extra keyword arguments for specific subclasses to
                implement
        Returns a pandas.DataFrame with (at least) the columns:
            gameid - id of the game which is being predicted
            chain - chain number of the sample
            draw - draw number within the chain
            homegoals - home goals, if predicted by model (NA otherwise)
            awaygoals - away goals, if predicted by model (NA otherwise)
            result - match result ('H','A','D')
        """
        # Sort out keyword arguments
        if (gameids == "all") or (gameids is None):
            gameids = itertools.chain(self.fitgameids, self.predictgameids)
        elif gameids == "fit":
            gameids = self.fitgameids
        elif gameids == "predict":
            gameids = self.predictgameids
        # Predict each game, assign gameid, and concatenate
        return pandas.concat(
            (self._predict(g, **kwargs).assign(gameid=g) for g in gameids),
            ignore_index=True)
    
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
        # Fill default pars and map to stan names
        if pars is None:
            pars = self._efl2stan.keys()
        # Get the data for autocorr
        samples = self.to_dataframe(pars, diagnostics=False, permuted=False)
        # Create blank df and compute autocorr
        ac = pandas.DataFrame(numpy.NaN, columns=pars, index=lags)
        for p in pars:
            # Can this be done faster with numpy.correlate?
            for n in lags:
                ac.loc[n,p] = samples[p].autocorr(n)
        return ac
    
    def traceplot(self, pars=None, combine_chains=False, page=(2,2)):
        """Create a traceplot of the samples from this model.
        Parameters:
            pars - list of parameters to make plots for. Default all.
            combine_chains - True: combine chains and plot one series per
                graph. False: plot each chain as individual series on each
                graph.
            page - tuple describing how many plots to put in one figure, 
                arranged in (rows, columns)
        Returns: a list of matplotlib.pyplot figures, one for each page.        
        """
        # Fill default pars and map to stan names
        if pars is None:
            pars = self._efl2stan.keys()
        # Get the data for plotting
        samples = self.to_dataframe(pars, diagnostics=False, permuted=False)
        # Create the figures and axes for plotting
        figs, axes = _make_axes(len(pars), page, "Traceplot")
        # Draw traceplots on all the axes objects
        for ax,p in zip(axes, pars):
            ax.set_title(p)
            if combine_chains:
                ax.plot(range(len(samples[p])), samples[p], linewidth=1)
            else:
                nchains = samples['chain'].max() + 1
                for c in samples['chain'].unique():
                    chain = samples[samples['chain'] == c]
                    ax.plot(range(len(chain[p])), chain[p], linewidth=1/nchains)
        # Return the list of figures
        return figs
    
    def densplot(self, pars=None, combine_chains=False, page=(2,2)):
        """Create a density plot of the samples from this model.
        Parameters:
            pars - list of parameters to make plots for. Default all.
            combine_chains - True: combine chains and plot one series per
                graph. False: plot each chain as individual series on each
                graph.
            page - tuple describing how many plots to put in one figure, 
                arranged in (rows, columns)
        Returns: a list of matplotlib.pyplot figures, one for each page.
        """
        # Fill default pars and map to stan names
        if pars is None:
            pars = self._efl2stan.keys()
        # Get the data for plotting
        samples = self.to_dataframe(pars, diagnostics=False, permuted=False)
        # Create the figures and axes for plotting
        figs, axes = _make_axes(len(pars), page, "Density Plot")
        # Draw density estimates on all of the axes
        for ax,p in zip(axes,pars):
            ax.set_title(p)
            if combine_chains:
                _draw_densplot(ax, samples[p])
            else:
                for c in samples['chain'].unique():
                    chain = samples[samples['chain'] == c]
                    _draw_densplot(ax, chain[p])
        # Return the list of figures
        return figs
    
    def boxplot(self, pars=None, vert=True):
        """Create side-by-side boxplots of the samples from this model.
        Parameters:
            pars - list of parameters to make plots for. Default all.
            vert - should the boxplots be vertical?
        Returns: a matplotlib.pyplot figure
        """
        # Fill default pars and map to stan names
        if pars is None:
            pars = self._efl2stan.keys()
        # Get the data for plotting
        samples = self.to_dataframe(pars, diagnostics=False, permuted=False)
        # Create the figures and axes for plotting
        figs, axes = _make_axes(1, (1,1), None)
        # Draw the boxplots on these axes
        axes[0].set_title("Parameter Boxplots")
        axes[0].boxplot(samples[pars].T, labels=pars, 
                        showmeans=True, showcaps=False, showfliers=False,
                        vert=vert)
        if vert:
            for tick in axes[0].get_xticklabels():
                tick.set_rotation(45)
        # Return the figure (no list, since boxplot always is on one figure)
        return figs[0]
    
    def acfplot(self, pars=None, maxlag=20, page=(2,2)):
        """Create an acf plot for the posterior samples of this model.
        Parameters:
            pars - the parameters to plot autocorrelation for. Default all.
            maxlag - the maximum lag to compute and plot autocorelation for.
            page - tuple describing how many plots to put in one figure, 
                arranged in (rows, columns)
        """
        # Find autocorrelations. Let autocorr do parameter parsing, and just
        # look at what columns it returns
        ac = self.autocorr(pars, range(maxlag+1))
        pars = list(ac.columns)
        # Create the figures and axes for plotting
        figs, axes = _make_axes(len(pars), page, None)
        # Draw the autocorrelation plots
        for ax,p in zip(axes, pars):
            ax.set_title(p)
            ax.bar(ac.index, ac[p])
        return figs
        


###############################################################################
## HELPER MODELS ##############################################################
###############################################################################


class EFL_GoalModel(EFLModel):
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
        samples['homegoals'] = samples[hg]
        samples['awaygoals'] = samples[ag]
        samples['result'] = 'D'
        samples.loc[(samples['homegoals']>samples['awaygoals']),'result'] = 'H'
        samples.loc[(samples['homegoals']<samples['awaygoals']),'result'] = 'A'
        # Drop quantity and return
        return samples.drop([hg, ag], axis=1)


class EFL_ResultModel(EFLModel):
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


###############################################################################
## UTILITY FUNCTIONS ##########################################################
###############################################################################
        

def _make_axes(num_plots, page, main_title):
    """Create figures and plots to be used by the plotting functions of
    EFLModel.
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
    if main_title is not None:
        for i,f in enumerate(figs):
            f.suptitle("{} {}/{}".format(main_title, i+1, numfigs))
            f.subplots_adjust(hspace=0.5, wspace=0.25)
    return figs, axes

def _draw_densplot(ax, data, nout=220):
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
    y = density(x)
    ax.plot(x,y)