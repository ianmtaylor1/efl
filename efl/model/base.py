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
                 chains=4, iter=10000, warmup=None, thin=1, n_jobs=1,
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
    
    def summary(self, pars=None, **kwargs):
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

