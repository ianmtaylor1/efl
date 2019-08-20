# -*- coding: utf-8 -*-
"""
eflmodels.py 

Module contains code to build and sample from EFL models.
"""

from . import cache

import numpy
import pandas
import itertools

class _EFLModel(object):
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
    
    Subclasses of EFLModel should:
        1. have a class-level attribute called _modelfile
        2. Implement _stan_inits method for generating initial values 
                from chain_id
        3. Have object-level attributes _efl2stan and _stan2efl for parameter
                name mapping between Stan output and EFL-relevant names.
        4. Provide a method _predict() for predicting a single game outcome.
    """
    
    def __init__(self, modeldata, fitgameids, predictgameids,
                 chains=4, iter=10000, warmup=None, thin=1, n_jobs=1):
        """Initialize the base properties of this model.
        Parameters:
            modeldata - Data that is passed to sampling(data=modeldata). Is
                also stored as self._modeldata for use in other methods.
            fitgameids, predictgameids - lists of game ids which were used to
                fit the model, or which are predicted by the model.
            chains, iter, warmup, thin, n_jobs - same as pystan options.
                Passed to sampling()
        """
        # Get model from compiled model cache
        self._model = cache.get_model(self._modelfile)
        # Store the data that was passed as an instance attribute
        self._modeldata = modeldata
        # Save the fit and predict game id's
        self.fitgameids = fitgameids
        self.predictgameids = predictgameids
        # Fit the model
        self.stanfit = self._model.sampling(
            data=self._modeldata, init=self._stan_inits,
            chains=chains, iter=iter, warmup=warmup, thin=thin, n_jobs=n_jobs)
    
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
    
    # Methods provided by this base class
    
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
                    - max(len(p) for p in self._stan2efl.keys())
        for (stanpar, eflpar) in self._stan2efl.items():
            spaces = addlength - (len(eflpar) - len(stanpar))
            if spaces >= 0: # Need to net insert spaces
                old = stanpar
                new = '{}{}'.format(eflpar, " "*spaces)
            else: # Need to net remove spaces
                old = '{}{}'.format(stanpar, " "*abs(spaces))
                new = eflpar
            sts = sts.replace(old, new)
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


class EFLSymOrdReg(_EFLModel):
    """*Sym*metric *Ord*inal *Reg*ression model for EFL data."""
    
    # Which Stan model file to use
    _modelfile = 'symordreg'
    
    def __init__(self, eflgames, **kwargs):
        modeldata, self._reference = self._get_model_data(eflgames)
        # TODO: get prior from previous fit or another way
        P = modeldata['P']
        modeldata['beta_prior_mean'] = numpy.zeros(P)
        modeldata['beta_prior_var'] = numpy.identity(P) * ((P/2)**2)
        modeldata['theta_prior_loc'] = 0
        modeldata['theta_prior_scale'] = 1
        # Call the superclass to fit the model
        super().__init__(
                modeldata,
                [g.id for g in eflgames.fit], # fitgameids
                [g.id for g in eflgames.predict], # predictgameids
                **kwargs)
        # Create parameter mappings.
        self._efl2stan = {'DrawBoundary':'theta', 'HomeField':'beta[1]'}
        for i,t in enumerate(eflgames.teams[1:], start=1):
            self._efl2stan[t.shortname] = 'beta[{}]'.format(i+1)
        self._stan2efl = dict(reversed(i) for i in self._efl2stan.items())
    
    @staticmethod
    def _get_model_data(games):
        """Take an EFLGames instance and transform it into a dict appropriate
        for the symordreg Stan model. Also returns the name of the reference
        team."""
        N = len(games.fit)
        N_new = len(games.predict)
        P = len(games.teams)
        Y = numpy.array([2 for g in games.fit], dtype=numpy.int_)
        Y[[(g.result.homegoals > g.result.awaygoals) for g in games.fit]] = 3
        Y[[(g.result.homegoals < g.result.awaygoals) for g in games.fit]] = 1
        X = numpy.zeros(shape=[N,P])
        X[:,0] = 1 # Homefield
        X_new = numpy.zeros(shape=[N_new,P])
        X_new[:,0] = 1 # Homefield
        for i,t in enumerate(games.teams[1:], start=1):
            X[[g.hometeamid == t.id for g in games.fit], i] = 1
            X[[g.awayteamid == t.id for g in games.fit], i] = -1
            X_new[[g.hometeamid == t.id for g in games.predict], i] = 1
            X_new[[g.awayteamid == t.id for g in games.predict], i] = -1
        return {'N':N, 'N_new':N_new, 'P':P, 'Y':Y, 'X':X, 'X_new':X_new}, games.teams[0].shortname
    
    def _stan_inits(self, chain_id=None):
        """Draw from a multivariate normal distribution and a logistic
        distribution to produce prior values for beta and theta."""
        beta = numpy.random.multivariate_normal(
                self._modeldata['beta_prior_mean'], 
                self._modeldata['beta_prior_var'])
        theta = abs(numpy.random.logistic(
                self._modeldata['theta_prior_loc'], 
                self._modeldata['theta_prior_scale']))
        return {'beta':beta, 'theta':theta}
    
    def _predict(self, gameid):
        """Predict the result for the game 'gameid' from this model's fitted 
        data."""
        return None
    
    def summary(self, pars=None, **kwargs):
        """Decorate the default summary. If pars is left as default, or
        includes the reference team, the reference is printed below all other 
        parameters."""
        if (pars is None) or (self._reference in pars):
            addref = True
        else:
            addref = False
        sts = super().summary(pars, **kwargs)
        if addref:
            newline = '**Reference: {} = 0'.format(self._reference)
            lines = sts.split("\n")
            # What is the last line that contains the name of a parameter?
            haspar = [any((p in ln) for p in self._efl2stan.keys()) for ln in lines]
            lastpar = max(i for i in range(len(haspar)) if haspar[i])
            # Inser the holdout right after that.
            lines.insert(lastpar + 1, newline)
            sts = "\n".join(lines)
        return sts
    
    def to_dataframe(self, pars=None, **kwargs):
        """Decorate the default to_dataframe. If pars is left as default, or
        includes the reference team, the reference is added as a column of
        all zeros."""
        if (pars is None) or (self._reference in pars):
            addref = True
        else:
            addref = False
        df = super().to_dataframe(pars, **kwargs)
        if addref:
            ispar = [(c in self._efl2stan.keys()) for c in df.columns]
            lastpar = max(i for i in range(len(ispar)) if ispar[i])
            df.insert(lastpar + 1, column=self._reference, value=0)
        return df
