# -*- coding: utf-8 -*-
"""
eflmodels.py 

Module contains code to build and sample from EFL models.
"""

from . import cache

import numpy
import pandas
import itertools
import re


###############################################################################
## BASE MODEL #################################################################
###############################################################################


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


class _EFL_GoalModel(_EFLModel):
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


class _EFL_ResultModel(_EFLModel):
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
## END-USER MODELS ############################################################
###############################################################################


class EFLSymOrdReg(_EFL_ResultModel):
    """*Sym*metric *Ord*inal *Reg*ression model for EFL data."""
    
    def __init__(self, eflgames, **kwargs):
        """Parameters:
            eflgames - an EFLGames instance
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        # Create priors
        priors = {}
        priors['home_prior_mean']   = 0
        priors['home_prior_sd']     = 1.8138
        priors['theta_prior_loc']   = 0
        priors['theta_prior_scale'] = 1
        P = len(eflgames.teams)
        priors['teams_prior_mean']  = numpy.zeros(P)
        priors['teams_prior_var']   = numpy.identity(P) * ((P/2)**2)
        # Create parameter mapping
        efl2stan = {'DrawBoundary':'theta', 'HomeField':'home'}
        for i,t in enumerate(eflgames.teams):
            efl2stan[t.shortname] = 'teams[{}]'.format(i+1)
        # Call super init
        super().__init__(
                modelfile      = 'symordreg_v2',
                eflgames       = eflgames,
                extramodeldata = priors,
                efl2stan       = efl2stan,
                **kwargs)
    
    def _stan_inits(self, chain_id=None):
        """Draw from a multivariate normal distribution and a logistic
        distribution to produce prior values for beta and theta."""
        P = self._modeldata['teams_prior_mean'].shape[0]
        teams = numpy.random.multivariate_normal(
                self._modeldata['teams_prior_mean'], 
                self._modeldata['teams_prior_var'])
        teams_raw = (teams - teams.mean())[:(P-1)]
        home = numpy.random.normal(
                self._modeldata['home_prior_mean'],
                self._modeldata['home_prior_sd'])
        theta = abs(numpy.random.logistic(
                self._modeldata['theta_prior_loc'], 
                self._modeldata['theta_prior_scale']))
        return {'teams_raw':teams_raw, 'home':home, 'theta':theta}


class EFLPoisRegNumberphile(_EFL_GoalModel):
    """Poisson Regression model based on Numberphile video with Tony Padilla
    https://www.numberphile.com/videos/a-million-simulated-seasons
    """
    
    def __init__(self, eflgames, **kwargs):
        """Parameters:
            eflgames - an EFLGames instance
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        # Create priors
        priors = {}
        priors['log_home_goals_prior_mean'] = 0
        priors['log_home_goals_prior_sd']   = 1
        priors['log_away_goals_prior_mean'] = 0
        priors['log_away_goals_prior_sd']   = 1
        P = len(eflgames.teams)
        priors['homeoff_prior_mean'] = numpy.zeros(P)
        priors['homeoff_prior_var']  = numpy.identity(P)
        priors['homedef_prior_mean'] = numpy.zeros(P)
        priors['homedef_prior_var']  = numpy.identity(P)
        priors['awayoff_prior_mean'] = numpy.zeros(P)
        priors['awayoff_prior_var']  = numpy.identity(P)
        priors['awaydef_prior_mean'] = numpy.zeros(P)
        priors['awaydef_prior_var']  = numpy.identity(P)
        # Create parameter mapping
        efl2stan = {'HomeGoals':'log_home_goals', 'AwayGoals':'log_away_goals'}
        for i,t in enumerate(eflgames.teams):
            efl2stan[t.shortname+' HO'] = 'homeoff[{}]'.format(i+1)
            efl2stan[t.shortname+' HD'] = 'homedef[{}]'.format(i+1)
            efl2stan[t.shortname+' AO'] = 'awayoff[{}]'.format(i+1)
            efl2stan[t.shortname+' AD'] = 'awaydef[{}]'.format(i+1)
        # Call super init
        super().__init__(
                modelfile      = 'poisreg_numberphile',
                eflgames       = eflgames,
                extramodeldata = priors,
                efl2stan       = efl2stan,
                **kwargs)
    
    def _stan_inits(self, chain_id=None):
        """Sample from the prior to produce initial values for each chain."""
        P = self._modeldata['homeoff_prior_mean'].shape[0]
        log_home_goals = numpy.random.normal(
                self._modeldata['log_home_goals_prior_mean'],
                self._modeldata['log_home_goals_prior_sd'])
        log_away_goals = numpy.random.normal(
                self._modeldata['log_away_goals_prior_mean'],
                self._modeldata['log_away_goals_prior_sd'])
        homeoff = numpy.random.multivariate_normal(
                self._modeldata['homeoff_prior_mean'],
                self._modeldata['homeoff_prior_var'])
        homedef = numpy.random.multivariate_normal(
                self._modeldata['homedef_prior_mean'],
                self._modeldata['homedef_prior_var'])
        awayoff = numpy.random.multivariate_normal(
                self._modeldata['awayoff_prior_mean'],
                self._modeldata['awayoff_prior_var'])
        awaydef = numpy.random.multivariate_normal(
                self._modeldata['awaydef_prior_mean'],
                self._modeldata['awaydef_prior_var'])
        homeoff_raw = (homeoff - homeoff.mean())[:(P-1)]
        homedef_raw = (homedef - homedef.mean())[:(P-1)]
        awayoff_raw = (awayoff - awayoff.mean())[:(P-1)]
        awaydef_raw = (awaydef - awaydef.mean())[:(P-1)]
        return {'log_home_goals':log_home_goals, 
                'log_away_goals':log_away_goals,
                'homeoff_raw':homeoff_raw,
                'homedef_raw':homedef_raw,
                'awayoff_raw':awayoff_raw,
                'awaydef_raw':awaydef_raw}
    

class EFLPoisRegSimple(_EFL_GoalModel):
    """Poisson Regression model based on Numberphile video with Tony Padilla
    https://www.numberphile.com/videos/a-million-simulated-seasons
    **But simplified, by assuming equal homefield advantage for all teams.
    """
    
    def __init__(self, eflgames, **kwargs):
        """Parameters:
            eflgames - an EFLGames instance
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        # Create priors
        priors = {}
        priors['log_home_goals_prior_mean'] = 0
        priors['log_home_goals_prior_sd']   = 1
        priors['log_away_goals_prior_mean'] = 0
        priors['log_away_goals_prior_sd']   = 1
        P = len(eflgames.teams)
        priors['offense_prior_mean'] = numpy.zeros(P)
        priors['offense_prior_var']  = numpy.identity(P)
        priors['defense_prior_mean'] = numpy.zeros(P)
        priors['defense_prior_var']  = numpy.identity(P)
        # Create parameter mapping
        efl2stan = {'HomeGoals':'log_home_goals', 'AwayGoals':'log_away_goals'}
        for i,t in enumerate(eflgames.teams):
            efl2stan[t.shortname+' Off'] = 'offense[{}]'.format(i+1)
            efl2stan[t.shortname+' Def'] = 'defense[{}]'.format(i+1)
        # Call super init
        super().__init__(
                modelfile      = 'poisreg_simple',
                eflgames       = eflgames,
                extramodeldata = priors,
                efl2stan       = efl2stan,
                **kwargs)
    
    def _stan_inits(self, chain_id=None):
        """Sample from the prior to produce initial values for each chain."""
        P = self._modeldata['offense_prior_mean'].shape[0]
        log_home_goals = numpy.random.normal(
                self._modeldata['log_home_goals_prior_mean'],
                self._modeldata['log_home_goals_prior_sd'])
        log_away_goals = numpy.random.normal(
                self._modeldata['log_away_goals_prior_mean'],
                self._modeldata['log_away_goals_prior_sd'])
        offense = numpy.random.multivariate_normal(
                self._modeldata['offense_prior_mean'],
                self._modeldata['offense_prior_var'])
        defense = numpy.random.multivariate_normal(
                self._modeldata['defense_prior_mean'],
                self._modeldata['defense_prior_var'])
        offense_raw = (offense - offense.mean())[:(P-1)]
        defense_raw = (defense - defense.mean())[:(P-1)]
        return {'log_home_goals':log_home_goals, 
                'log_away_goals':log_away_goals,
                'offense_raw':offense_raw,
                'defense_raw':defense_raw}


class EFLPoisRegHier(_EFL_GoalModel):
    """Class for a poisson regression model where the team's 'sub-parameters'
    e.g. home offense, away defense, etc. are hierarchically determined by a
    latent team strength parameter."""
    
    def __init__(self, eflgames, **kwargs):
        """Parameters:
            eflgames - an EFLGames instance
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        # Create priors
        priors = {}
        priors['log_goals_prior_mean'] = 0
        priors['log_goals_prior_sd']   = 1
        priors['home_prior_mean'] = 0
        priors['home_prior_sd']   = 1
        priors['sigma2_prior_alpha'] = 1
        priors['sigma2_prior_beta']  = 1
        priors['s2home_prior_alpha'] = 1
        priors['s2home_prior_beta']  = 1
        priors['rho_prior_alpha'] = 1
        priors['rho_prior_beta']  = 1
        P = len(eflgames.teams)
        priors['teams_prior_mean'] = numpy.zeros(P)
        priors['teams_prior_var']  = numpy.identity(P)
        # Create parameter mapping
        efl2stan = {'AvgGoals':'log_goals', 'HomeField':'home',
                    'TeamSkillVar':'sigma2', 'HomeVar':'s2home', 
                    'TeamSkillCorr':'rho'}
        for i,t in enumerate(eflgames.teams):
            efl2stan[t.shortname] = 'teams[{}]'.format(i+1)
            efl2stan[t.shortname+' HO'] = 'homeoff[{}]'.format(i+1)
            efl2stan[t.shortname+' HD'] = 'homedef[{}]'.format(i+1)
            efl2stan[t.shortname+' AO'] = 'awayoff[{}]'.format(i+1)
            efl2stan[t.shortname+' AD'] = 'awaydef[{}]'.format(i+1)
        # Call super init
        super().__init__(
                modelfile      = 'poisreg_hier',
                eflgames       = eflgames,
                extramodeldata = priors,
                efl2stan       = efl2stan,
                **kwargs)
    
    def _stan_inits(self, chain_id=None):
        """Sample from the prior to produce starting values for each chain."""
        # Baseline goals and homefield advantage
        log_goals = numpy.random.normal(
                self._modeldata['log_goals_prior_mean'],
                self._modeldata['log_goals_prior_sd'])
        home = numpy.random.normal(
                self._modeldata['home_prior_mean'],
                self._modeldata['home_prior_sd'])
        # Variance parameters for team sub-parameters
        sigma2 = numpy.random.gamma(
                self._modeldata['sigma2_prior_alpha'],
                1/self._modeldata['sigma2_prior_beta'])
        s2home = numpy.random.gamma(
                self._modeldata['s2home_prior_alpha'],
                1/self._modeldata['s2home_prior_beta'])
        rho = numpy.random.beta(
                self._modeldata['rho_prior_alpha'],
                self._modeldata['rho_prior_beta']) * 2 - 1
        # Individual team strengths
        teams = numpy.random.multivariate_normal(
                self._modeldata['teams_prior_mean'],
                self._modeldata['teams_prior_var'])
        teams = teams - teams.mean()
        # Team subparameters
        P = self._modeldata['teams_prior_mean'].shape[0]
        teamvar = numpy.array(
                [[sigma2+s2home, rho*sigma2, s2home,        0          ],
                 [rho*sigma2,    sigma2,     0,             0          ],
                 [s2home,        0,          sigma2+s2home, rho*sigma2 ],
                 [0,             0,          rho*sigma2,    sigma2     ]])
        homeoff = numpy.zeros(P)
        homedef = numpy.zeros(P)
        awayoff = numpy.zeros(P)
        awaydef = numpy.zeros(P)
        for i in range(P):
            mean = numpy.array([home, 0, home, 0]) + teams[i]
            x = numpy.random.multivariate_normal(mean, teamvar)
            homeoff[i] = x[0]
            awayoff[i] = x[1]
            homedef[i] = x[2]
            awaydef[i] = x[3]
        # Put together and return
        return {'log_goals':log_goals, 'home':home, 'teams_raw':teams[:(P-1)],
                'sigma2':sigma2, 's2home':s2home, 'rho':rho,
                'homeoff':homeoff, 'homedef':homedef,
                'awayoff':awayoff, 'awaydef':awaydef}


###############################################################################
## OLD END-USER MODELS ########################################################
###############################################################################


class EFLSymOrdReg_Old(_EFL_WithReference, _EFLModel):
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
        samples['homegoals'] = numpy.NaN
        samples['awaygoals'] = numpy.NaN
        # Drop quantity and return
        return samples.drop(qtyname, axis=1)


class EFLPoisRegNumberphile_Old(_EFL_WithReference, _EFLModel):
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


class EFLPoisRegSimple_Old(_EFL_WithReference, _EFLModel):
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


class EFLPoisRegHier_Old(_EFL_WithReference, _EFLModel):
    """Class for a poisson regression model where the team's 'sub-parameters'
    e.g. home offense, away defense, etc. are hierarchically determined by a
    latent team strength parameter."""
    
    def __init__(self,eflgames, **kwargs):
        modeldata = self._get_model_data(eflgames)
        reference = eflgames.teams[-1].shortname
        # TODO: get prior from previous fit or another way
        nT = modeldata['nTeams']
        modeldata['teams_prior_mean'] = numpy.zeros(nT-1)
        modeldata['teams_prior_var'] = numpy.identity(nT-1)*2
        modeldata['teamvar_prior_nu'] = 3
        modeldata['teamvar_prior_Sigma'] = numpy.identity(2)/6
        modeldata['log_goals_prior_mean'] = 0
        modeldata['log_goals_prior_sd'] = numpy.sqrt(2)
        modeldata['home_prior_mean'] = 0
        modeldata['home_prior_sd'] = 1
        # Parameter mappings
        efl2stan = {'Log Goals':'log_goals', 'Homefield':'home'}
        for i,t in enumerate(eflgames.teams[:-1]):
            efl2stan[t.shortname] = 'teams[{}]'.format(i+1)
        for i,t in enumerate(eflgames.teams):
            efl2stan[t.shortname+' HO'] = 'beta[{}]'.format(4*i+1)
            efl2stan[t.shortname+' AO'] = 'beta[{}]'.format(4*i+2)
            efl2stan[t.shortname+' HD'] = 'beta[{}]'.format(4*i+3)
            efl2stan[t.shortname+' AD'] = 'beta[{}]'.format(4*i+4)
        efl2stan['HomeTeamVar'] = 'teamvar[1,1]'
        efl2stan['AwayTeamVar'] = 'teamvar[2,2]'
        efl2stan['TeamODCov'] = 'teamvar[1,2]'
        super().__init__(
                modelfile      = 'poisreg_hier_team',
                modeldata      = modeldata,
                fitgameids     = [g.id for g in eflgames.fit],
                predictgameids = [g.id for g in eflgames.predict],
                efl2stan       = efl2stan,
                references     = [reference],
                **kwargs)
        # TODO: predict quantities
    
    @staticmethod
    def _get_model_data(games):
        """Take an EFLGames instance and transform it into a dict appropriate
        for the poisreg_hier_team Stan model."""
        # Number of games and teams
        nGames = len(games.fit)
        nGames_new = len(games.predict)
        nTeams = len(games.teams)
        # Team indexes for each game
        teamidxmap = {t.id:(i+1) for i,t in enumerate(games.teams)}
        hometeamidx = numpy.array(
                [teamidxmap[g.hometeamid] for g in games.fit], 
                dtype=numpy.int_)
        awayteamidx = numpy.array(
                [teamidxmap[g.awayteamid] for g in games.fit], 
                dtype=numpy.int_)
        hometeamidx_new = numpy.array(
                [teamidxmap[g.hometeamid] for g in games.predict], 
                dtype=numpy.int_)
        awayteamidx_new = numpy.array(
                [teamidxmap[g.awayteamid] for g in games.predict], 
                dtype=numpy.int_)
        # Game goals
        homegoals = numpy.array(
                [g.result.homegoals for g in games.fit],
                dtype=numpy.int_)
        awaygoals = numpy.array(
                [g.result.awaygoals for g in games.fit],
                dtype=numpy.int_)
        # Return the model data
        return {'nGames':nGames, 'nGames_new':nGames_new, 'nTeams':nTeams,
                'hometeamidx':hometeamidx, 'awayteamidx':awayteamidx,
                'homegoals':homegoals, 'awaygoals':awaygoals,
                'hometeamidx_new':hometeamidx_new, 
                'awayteamidx_new':awayteamidx_new}
        