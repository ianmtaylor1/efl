# -*- coding: utf-8 -*-
"""
eflmodels.py 

Module contains code to build and sample from EFL models.
"""

from . import cache

import numpy
import pandas
import itertools


#######################################
## BASE MODEL #########################
#######################################


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
        8. A readable attribute parameters, which is a list of all efl (human-
                readable) parameters available in the model.
    
    Subclasses of EFLModel should:
        1. Implement _stan_inits method for generating initial values 
                from chain_id
        2. Provide a method _predict() for predicting a single game outcome.
    """
    
    def __init__(self, modelfile, modeldata, fitgameids, predictgameids,
                 efl2stan,
                 chains=4, iter=10000, warmup=None, thin=1, n_jobs=1):
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
    
    # Methods and properties provided by this base class
    
    @property
    def parameters(self):
        return self._efl2stan.keys()
    
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


#######################################
## HELPER MODELS ######################
#######################################


class _EFL_WithReference(_EFLModel):
    """Mix in class for models which have 'reference's or 'holdout's,
    parameters which are set equal to zero. Wraps the summary and to_dataframe
    methods to include the reference parameters."""
    
    def __init__(self, references=[], **kwargs):
        """References = list of parameters which are references."""
        self._references = list(references)
        super().__init__(**kwargs)
    
    def summary(self, pars=None, **kwargs):
        """Decorate the default summary. If pars is left as default, or
        includes the reference team, the reference is printed below all other 
        parameters."""
        if pars is None:
            addref = self._references
        else:
            addref = list(set(pars) & set(self._references))
        sts = super().summary(pars, **kwargs)
        if len(addref) > 0:
            newline = '** Reference:'
            for ref in addref:
                newline += '\n{} = 0'.format(ref)
            lines = sts.split("\n")
            # What is the last line that contains the name of a parameter?
            haspar = [any((p in ln) for p in self.parameters) for ln in lines]
            lastpar = max(i for i in range(len(haspar)) if haspar[i])
            # Inser the holdout right after that.
            lines.insert(lastpar + 1, newline)
            sts = "\n".join(lines)
        return sts
    
    def to_dataframe(self, pars=None, **kwargs):
        """Decorate the default to_dataframe. If pars is left as default, or
        includes the reference team, the reference is added as a column of
        all zeros."""
        if pars is None:
            addref = self._references
        else:
            addref = list(set(pars) & set(self._references))
        df = super().to_dataframe(pars, **kwargs)
        if len(addref) > 0:
            ispar = [(c in self.parameters) for c in df.columns]
            lastpar = max(i for i in range(len(ispar)) if ispar[i])
            for ref in addref:
                df.insert(lastpar + 1, column=ref, value=0)
        return df


#######################################
## END-USER MODELS ####################
#######################################


class EFLSymOrdReg(_EFL_WithReference, _EFLModel):
    """*Sym*metric *Ord*inal *Reg*ression model for EFL data."""
    
    def __init__(self, eflgames, **kwargs):
        modeldata, reference = self._get_model_data(eflgames)
        # TODO: get prior from previous fit or another way
        P = modeldata['P']
        modeldata['beta_prior_mean'] = numpy.zeros(P)
        modeldata['beta_prior_var'] = numpy.identity(P) * ((P/2)**2)
        modeldata['theta_prior_loc'] = 0
        modeldata['theta_prior_scale'] = 1
        # Create parameter mappings.
        efl2stan = {'DrawBoundary':'theta', 'HomeField':'beta[1]'}
        for i,t in enumerate(eflgames.teams[1:], start=1):
            efl2stan[t.shortname] = 'beta[{}]'.format(i+1)
        # Call the superclass to fit the model
        super().__init__(
                modelfile      = 'symordreg',
                modeldata      = modeldata,
                fitgameids     = [g.id for g in eflgames.fit],
                predictgameids = [g.id for g in eflgames.predict],
                efl2stan       = efl2stan,
                references     = [reference],
                **kwargs)
        # Create mappings from gameids to post. pred. sampled quantities
        self._predictqtys = {}
        for i,g in enumerate(eflgames.fit):
            self._predictqtys[g.id] = 'Y_pred[{}]'.format(i+1)
        for i,g in enumerate(eflgames.predict):
            self._predictqtys[g.id] = 'Y_new_pred[{}]'.format(i+1)
    
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
        # Find the quantity we need to look at
        qtyname = self._predictqtys[gameid]
        # Pull that quantity
        samples = self.stanfit.to_dataframe(pars=[qtyname], permuted=False,
                                            diagnostics=False)
        # Map to a result
        samples['result'] = samples[qtyname].apply(
                lambda x: ['A','D','H'][int(x)-1])
        # Drop quantity and return
        return samples.drop(qtyname, axis=1)


class EFLPoisRegNumberphile(_EFL_WithReference, _EFLModel):
    """Poisson Regression model based on Numberphile video with Tony Padilla
    https://www.numberphile.com/videos/a-million-simulated-seasons
    """
    
    def __init__(self, eflgames, **kwargs):
        modeldata, ref = self._get_model_data(eflgames)
        references = [ref+' HO', ref+' HD', ref+' AO', ref+' AD']
        # TODO: get prior from previous fit or another way
        P = modeldata['P']
        modeldata['beta_prior_mean'] = numpy.zeros(P)
        modeldata['beta_prior_var'] = numpy.identity(P) * 2
        # Create parameter mappings.
        efl2stan = {'HomePoints':'beta[1]', 'AwayPoints':'beta[2]'}
        for i,t in enumerate(eflgames.teams[1:], start=1):
            efl2stan[t.shortname+' HO'] = 'beta[{}]'.format((4*i)-1)
            efl2stan[t.shortname+' HD'] = 'beta[{}]'.format((4*i+1)-1)
            efl2stan[t.shortname+' AO'] = 'beta[{}]'.format((4*i+2)-1)
            efl2stan[t.shortname+' AD'] = 'beta[{}]'.format((4*i+3)-1)
        # Call the superclass to fit the model
        super().__init__(
                modelfile      = 'poisreg',
                modeldata      = modeldata,
                fitgameids     = [g.id for g in eflgames.fit],
                predictgameids = [g.id for g in eflgames.predict],
                efl2stan       = efl2stan,
                references     = references,
                **kwargs)
        # Create mappings from gameids to post. pred. sampled quantities
        self._predictqtys = {}
        for i,g in enumerate(eflgames.fit):
            self._predictqtys[g.id] = ('Y_pred[{}]'.format(2*i+1),
                                       'Y_pred[{}]'.format(2*i+2))
        for i,g in enumerate(eflgames.predict):
            self._predictqtys[g.id] = ('Y_new_pred[{}]'.format(2*i+1),
                                       'Y_new_pred[{}]'.format(2*i+2))
    
    @staticmethod
    def _get_model_data(games):
        """Take an EFLGames instance and transform it into a dict appropriate
        for the poisreg Stan model. Also returns the name of the reference
        team."""
        N = len(games.fit) * 2
        N_new = len(games.predict) * 2
        P = len(games.teams) * 4 - 2
        Y = numpy.zeros(N, dtype=numpy.int_)
        for i,g in enumerate(games.fit):
            Y[2*i] = g.result.homegoals
            Y[2*i+1] = g.result.awaygoals
        # X - fill entries by team
        X = numpy.zeros(shape=[N,P])
        X[range(0,N,2),0] = 1 # HomePoints
        X[range(1,N,2),1] = 1 # AwayPoints
        for i,t in enumerate(games.teams[1:], start=1):
            ishome = numpy.array([j for j,g in enumerate(games.fit) if g.hometeamid == t.id], dtype=numpy.int_)
            isaway = numpy.array([j for j,g in enumerate(games.fit) if g.awayteamid == t.id], dtype=numpy.int_)
            X[2*ishome,     (4*i) - 2]     = 1  # Home Offense
            X[2*ishome + 1, (4*i + 1) - 2] = -1 # Home Defense
            X[2*isaway + 1, (4*i + 2) - 2] = 1  # Away Offense
            X[2*isaway,     (4*i + 3) - 2] = -1 # Away Defense
        # X_new - fill entries by team
        X_new = numpy.zeros(shape=[N_new,P])
        X_new[range(0,N_new,2),0] = 1 # HomePoints
        X_new[range(1,N_new,2),1] = 1 # AwayPoints
        for i,t in enumerate(games.teams[1:], start=1):
            ishome = numpy.array([j for j,g in enumerate(games.predict) if g.hometeamid == t.id], dtype=numpy.int_)
            isaway = numpy.array([j for j,g in enumerate(games.predict) if g.awayteamid == t.id], dtype=numpy.int_)
            X_new[2*ishome,     (4*i) - 2]     = 1  # Home Offense
            X_new[2*ishome + 1, (4*i + 1) - 2] = -1 # Home Defense
            X_new[2*isaway + 1, (4*i + 2) - 2] = 1  # Away Offense
            X_new[2*isaway,     (4*i + 3) - 2] = -1 # Away Defense
        return {'N':N, 'N_new':N_new, 'P':P, 'Y':Y, 'X':X, 'X_new':X_new}, games.teams[0].shortname
    
    def _stan_inits(self, chain_id=None):
        """Draw from a multivariate normal distribution to produce prior
        values for beta."""
        beta = numpy.random.multivariate_normal(
                self._modeldata['beta_prior_mean'], 
                self._modeldata['beta_prior_var'])
        return {'beta':beta}
    
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


class EFLPoisRegSimple(_EFL_WithReference, _EFLModel):
    """Poisson Regression model based on Numberphile video with Tony Padilla
    https://www.numberphile.com/videos/a-million-simulated-seasons
    **But simplified, by assuming equal homefield advantage for all teams.
    """
    
    def __init__(self, eflgames, **kwargs):
        modeldata, ref = self._get_model_data(eflgames)
        references = [ref+' O', ref+' D']
        # TODO: get prior from previous fit or another way
        P = modeldata['P']
        modeldata['beta_prior_mean'] = numpy.zeros(P)
        modeldata['beta_prior_var'] = numpy.identity(P) * 2
        # Create parameter mappings.
        efl2stan = {'HomePoints':'beta[1]', 'AwayPoints':'beta[2]'}
        for i,t in enumerate(eflgames.teams[1:], start=1):
            efl2stan[t.shortname+' O'] = 'beta[{}]'.format((2*i)+1)
            efl2stan[t.shortname+' D'] = 'beta[{}]'.format((2*i+1)+1)
        # Call the superclass to fit the model
        super().__init__(
                modelfile      = 'poisreg',
                modeldata      = modeldata,
                fitgameids     = [g.id for g in eflgames.fit],
                predictgameids = [g.id for g in eflgames.predict],
                efl2stan       = efl2stan,
                references     = references,
                **kwargs)
        # Create mappings from gameids to post. pred. sampled quantities
        self._predictqtys = {}
        for i,g in enumerate(eflgames.fit):
            self._predictqtys[g.id] = ('Y_pred[{}]'.format(2*i+1),
                                       'Y_pred[{}]'.format(2*i+2))
        for i,g in enumerate(eflgames.predict):
            self._predictqtys[g.id] = ('Y_new_pred[{}]'.format(2*i+1),
                                       'Y_new_pred[{}]'.format(2*i+2))
    
    @staticmethod
    def _get_model_data(games):
        """Take an EFLGames instance and transform it into a dict appropriate
        for the poisreg Stan model. Also returns the name of the reference
        team."""
        N = len(games.fit) * 2
        N_new = len(games.predict) * 2
        P = len(games.teams) * 2
        Y = numpy.zeros(N, dtype=numpy.int_)
        for i,g in enumerate(games.fit):
            Y[2*i] = g.result.homegoals
            Y[2*i+1] = g.result.awaygoals
        # X - fill entries by team
        X = numpy.zeros(shape=[N,P])
        X[range(0,N,2),0] = 1 # HomePoints
        X[range(1,N,2),1] = 1 # AwayPoints
        for i,t in enumerate(games.teams[1:], start=1):
            ishome = numpy.array([j for j,g in enumerate(games.fit) if g.hometeamid == t.id], dtype=numpy.int_)
            isaway = numpy.array([j for j,g in enumerate(games.fit) if g.awayteamid == t.id], dtype=numpy.int_)
            X[2*ishome,     (2*i)]     = 1  # Offense in the homegoals row
            X[2*ishome + 1, (2*i + 1)] = -1 # Defense in the awaygoals row
            X[2*isaway,     (2*i + 1)] = -1 # Defense in the homegoals row
            X[2*isaway + 1, (2*i)]     = 1  # Offense in the awaygoals row
        # X_new - fill entries by team
        X_new = numpy.zeros(shape=[N_new,P])
        X_new[range(0,N_new,2),0] = 1 # HomePoints
        X_new[range(1,N_new,2),1] = 1 # AwayPoints
        for i,t in enumerate(games.teams[1:], start=1):
            ishome = numpy.array([j for j,g in enumerate(games.predict) if g.hometeamid == t.id], dtype=numpy.int_)
            isaway = numpy.array([j for j,g in enumerate(games.predict) if g.awayteamid == t.id], dtype=numpy.int_)
            X_new[2*ishome,     (2*i)]     = 1  # Offense in the homegoals row
            X_new[2*ishome + 1, (2*i + 1)] = -1 # Defense in the awaygoals row
            X_new[2*isaway,     (2*i + 1)] = -1 # Defense in the homegoals row
            X_new[2*isaway + 1, (2*i)]     = 1  # Offense in the awaygoals row
        return {'N':N, 'N_new':N_new, 'P':P, 'Y':Y, 'X':X, 'X_new':X_new}, games.teams[0].shortname
    
    def _stan_inits(self, chain_id=None):
        """Draw from a multivariate normal distribution to produce prior
        values for beta."""
        beta = numpy.random.multivariate_normal(
                self._modeldata['beta_prior_mean'], 
                self._modeldata['beta_prior_var'])
        return {'beta':beta}
    
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
