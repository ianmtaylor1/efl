#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poisregnumberphile.py

Contains the PoisRegNumberphile model and associated other classes.
"""

from . import base
from .. import util

import numpy


class PoisRegNumberphile(base.GoalModel):
    """Poisson Regression model based on Numberphile video with Tony Padilla
    https://www.numberphile.com/videos/a-million-simulated-seasons
    """
    
    def __init__(self, eflgames, prior=None, **kwargs):
        """Parameters:
            eflgames - a Games instance
            prior - instance of PoisRegNumberphile_Prior, or None for 
                a diffuse prior
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        team_names = [t.shortname for t in eflgames.teams]
        # Create priors
        if prior is None:
            prior = PoisRegNumberphile_Prior.default_prior(team_names)
        # Create parameter mapping
        efl2stan = {'HomeGoals':'log_home_goals', 'AwayGoals':'log_away_goals'}
        for i,t in enumerate(team_names):
            efl2stan[t+' HO'] = 'homeoff[{}]'.format(i+1)
            efl2stan[t+' HD'] = 'homedef[{}]'.format(i+1)
            efl2stan[t+' AO'] = 'awayoff[{}]'.format(i+1)
            efl2stan[t+' AD'] = 'awaydef[{}]'.format(i+1)
        pargroups = {'goals':['HomeGoals','AwayGoals'],
                     'homeoff':[t+' HO' for t in team_names],
                     'homedef':[t+' HD' for t in team_names],
                     'awayoff':[t+' AO' for t in team_names],
                     'awaydef':[t+' AD' for t in team_names]}
        for t in team_names:
            pargroups[t] = [t+' HO', t+' HD', t+' AO', t+' AD']
        # Call super init
        super().__init__(
                modelfile      = 'poisreg_numberphile',
                eflgames       = eflgames,
                extramodeldata = prior.get_params(team_names),
                efl2stan       = efl2stan,
                pargroups      = pargroups,
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
    

class PoisRegNumberphile_Prior(object):
    """A class holding a prior for the PoisRegNumberphile model."""
    
    def __init__(self, homeoff_prior_mean, homeoff_prior_var, 
                 homedef_prior_mean, homedef_prior_var,
                 awayoff_prior_mean, awayoff_prior_var,
                 awaydef_prior_mean, awaydef_prior_var, 
                 team_names,
                 log_home_goals_prior_mean, log_home_goals_prior_sd,
                 log_away_goals_prior_mean, log_away_goals_prior_sd):
        """This constructor is pretty much never called in most typical uses.
        Parameters:
            homeoff_prior_mean - a 1d numpy array containing prior means of
                team home offense parameters. Centers it around 0.
            homeoff_prior_var - a 2d numpy array containing the prior covariance
                of team home offense parameters. Assumes it is symmetric pos-def, 
                does no checks.
            homedef_prior_mean - a 1d numpy array containing prior means of
                team home defense parameters. Centers it around 0.
            homedef_prior_var - a 2d numpy array containing the prior covariance
                of team home defense parameters. Assumes it is symmetric pos-def, 
                does no checks.
            awayoff_prior_mean - a 1d numpy array containing prior means of
                team away offense parameters. Centers it around 0.
            awayoff_prior_var - a 2d numpy array containing the prior covariance
                of team away offense parameters. Assumes it is symmetric pos-def, 
                does no checks.
            awaydef_prior_mean - a 1d numpy array containing prior means of
                team away defense parameters. Centers it around 0.
            awaydef_prior_var - a 2d numpy array containing the prior covariance
                of team away defense parameters. Assumes it is symmetric pos-def, 
                does no checks.
            team_names - a list of names used as labels for the indices of
                each of the previous four parameters. i.e. the names of the 
                elements in offense_prior_mean/defense_prior_mean and the rows
                and columns of offense_prior_var/defense_prior_var should be
                in the order of this list.
            log_home_goals_prior_mean, log_home_goals_prior_sd - prior
                parameters for the home goals scored parameter
            log_away_goals_prior_mean, log_away_goals_prior_sd - prior
                parameters for the away goals scored parameter
        """
        # Center mean around zero (due to the model's parameter centering)
        self._homeoff_prior_mean = homeoff_prior_mean - homeoff_prior_mean.mean()
        self._homedef_prior_mean = homedef_prior_mean - homedef_prior_mean.mean()
        self._awayoff_prior_mean = awayoff_prior_mean - awayoff_prior_mean.mean()
        self._awaydef_prior_mean = awaydef_prior_mean - awaydef_prior_mean.mean()
        # Accept the variance matrix as is (future enhancement?)
        self._homeoff_prior_var = homeoff_prior_var
        self._homedef_prior_var = homedef_prior_var
        self._awayoff_prior_var = awayoff_prior_var
        self._awaydef_prior_var = awaydef_prior_var
        # Turn the list into a dict of indices
        self._team_map = {t:i for i,t in enumerate(team_names)}
        # Copy all the other parameters
        self._log_home_goals_prior_mean = log_home_goals_prior_mean
        self._log_home_goals_prior_sd = log_home_goals_prior_sd
        self._log_away_goals_prior_mean = log_away_goals_prior_mean
        self._log_away_goals_prior_sd = log_away_goals_prior_sd
        
    def get_params(self, teams):
        """Get the stored prior parameters, but reordered by the order of the
        supplied team names. If any teams are absent from the supplied list,
        they will be dropped silently.
        Parameters:
            teams - a list of team names used for selecting and ordering the
            parameters while returning them.
        Returns:
            a dict in the format needed for pystan (and the EFLModel class)
        """
        idx = [self._team_map[t] for t in teams]
        return {'homeoff_prior_mean':self._homeoff_prior_mean[idx],
                'homeoff_prior_var':self._homeoff_prior_var[idx,:][:,idx],
                'homedef_prior_mean':self._homedef_prior_mean[idx],
                'homedef_prior_var':self._homedef_prior_var[idx,:][:,idx],
                'awayoff_prior_mean':self._awayoff_prior_mean[idx],
                'awayoff_prior_var':self._awayoff_prior_var[idx,:][:,idx],
                'awaydef_prior_mean':self._awaydef_prior_mean[idx],
                'awaydef_prior_var':self._awaydef_prior_var[idx,:][:,idx],
                'log_home_goals_prior_mean':self._log_home_goals_prior_mean,
                'log_home_goals_prior_sd':self._log_home_goals_prior_sd,
                'log_away_goals_prior_mean':self._log_away_goals_prior_mean,
                'log_away_goals_prior_sd':self._log_away_goals_prior_sd}
        
    # Class methods for creating instances through various methods
    
    @classmethod
    def default_prior(cls, team_names):
        """Instantiates a wide, default prior for this model."""
        P = len(team_names)
        return cls(homeoff_prior_mean = numpy.zeros(P),
                   homeoff_prior_var = numpy.identity(P),
                   homedef_prior_mean = numpy.zeros(P),
                   homedef_prior_var = numpy.identity(P),
                   awayoff_prior_mean = numpy.zeros(P),
                   awayoff_prior_var = numpy.identity(P),
                   awaydef_prior_mean = numpy.zeros(P),
                   awaydef_prior_var = numpy.identity(P),
                   team_names = team_names,
                   log_home_goals_prior_mean = 0,
                   log_home_goals_prior_sd = 1,
                   log_away_goals_prior_mean = 0,
                   log_away_goals_prior_sd = 1)
        
    @classmethod
    def from_fit(cls, fit, spread=1.0, regression=1.0,
                 team_spread=None, other_spread=None,
                 relegated_in=[], promoted_out=[],
                 promoted_in=[], relegated_out=[]):
        """Create a prior from the posterior of a previous PoisRegNumberphile fit.
        Parameters:
            fit - the previous instance of PoisRegNumberphile
            spread - factor by which to inflate variances of all parameters
                from the posterior of 'fit'. Think of this as season-to-season
                uncertainty.
            team_spread, other_spread - variance inflation factors for team
                strength (offense, defense) and other parameters, respectively.
                If supplied, overrides default from spread.
            regression - is multiplied by team offense/defense means
            relegated_in - list of team names who were relegated into this
                league from a higher league between the season of 'fit' and now
            promoted_out - list of team names who were promoted out of this
                league into a higher league between the season of 'fit' and now
            promoted_in - list of team names who were promoted into this
                league from a lower league between the season of 'fit' and now
            relegated_out - list of team names who were relegated out of this
                league into a lower league between the season of 'fit' and now
        """
        # Input checks
        if len(relegated_out) != len(promoted_in):
            raise Exception('len(relegated_out) must equal len(promoted_in)')
        if len(promoted_out) != len(relegated_in):
            raise Exception('len(promoted_out) must equal len(relegated_in)')
        if len(set(relegated_out) & set(promoted_out)) > 0:
            raise Exception('promoted_out and relegated_out cannot have common teams')
        if len(set(relegated_in) & set(promoted_in)) > 0:
            raise Exception('promoted_in and relegated_in cannot have common teams')
        # Default spreads
        if team_spread is None:
            team_spread = spread
        if other_spread is None:
            other_spread = spread
        # Get the posterior samples from the fit
        df = fit.to_dataframe(diagnostics=False)
        # Goals parameters
        log_home_goals_prior_mean = df['HomeGoals'].mean()
        log_home_goals_prior_sd = df['HomeGoals'].std() * numpy.sqrt(other_spread)
        log_away_goals_prior_mean = df['AwayGoals'].mean()
        log_away_goals_prior_sd = df['AwayGoals'].std() * numpy.sqrt(other_spread)
        # Promotion/relegation
        if len(promoted_out) > 0:
            util.shuffle_rename(df, [t+' HO' for t in promoted_out],
                                newnames=[t+' HO' for t in relegated_in])
            util.shuffle_rename(df, [t+' HD' for t in promoted_out],
                                newnames=[t+' HD' for t in relegated_in])
            util.shuffle_rename(df, [t+' AO' for t in promoted_out],
                                newnames=[t+' AO' for t in relegated_in])
            util.shuffle_rename(df, [t+' AD' for t in promoted_out],
                                newnames=[t+' AD' for t in relegated_in])
        if len(relegated_out) > 0:
            util.shuffle_rename(df, [t+' HO' for t in relegated_out],
                                newnames=[t+' HO' for t in promoted_in])
            util.shuffle_rename(df, [t+' HD' for t in relegated_out],
                                newnames=[t+' HD' for t in promoted_in])
            util.shuffle_rename(df, [t+' AO' for t in relegated_out],
                                newnames=[t+' AO' for t in promoted_in])
            util.shuffle_rename(df, [t+' AD' for t in relegated_out],
                                newnames=[t+' AD' for t in promoted_in])
        # Team offense/defense priors
        team_names = [c[:-3] for c in df.columns if c[-3:]==' HO']
        homeoff_prior_mean, homeoff_prior_var = util.mean_var(
                df, cols=[t+' HO' for t in team_names],
                meanregress=regression, varspread=team_spread, ridge=0.0001)
        homedef_prior_mean, homedef_prior_var = util.mean_var(
                df, cols=[t+' HD' for t in team_names],
                meanregress=regression, varspread=team_spread, ridge=0.0001)
        awayoff_prior_mean, awayoff_prior_var = util.mean_var(
                df, cols=[t+' AO' for t in team_names],
                meanregress=regression, varspread=team_spread, ridge=0.0001)
        awaydef_prior_mean, awaydef_prior_var = util.mean_var(
                df, cols=[t+' AD' for t in team_names],
                meanregress=regression, varspread=team_spread, ridge=0.0001)
        # Assemble and return
        return cls(homeoff_prior_mean = homeoff_prior_mean,
                   homeoff_prior_var = homeoff_prior_var,
                   homedef_prior_mean = homedef_prior_mean,
                   homedef_prior_var = homedef_prior_var,
                   awayoff_prior_mean = awayoff_prior_mean,
                   awayoff_prior_var = awayoff_prior_var,
                   awaydef_prior_mean = awaydef_prior_mean,
                   awaydef_prior_var = awaydef_prior_var,
                   team_names = team_names,
                   log_home_goals_prior_mean = log_home_goals_prior_mean,
                   log_home_goals_prior_sd = log_home_goals_prior_sd,
                   log_away_goals_prior_mean = log_away_goals_prior_mean,
                   log_away_goals_prior_sd = log_away_goals_prior_sd)

