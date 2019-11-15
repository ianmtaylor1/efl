#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poisregsimple.py

Contains the EFLPoisRegSimple model and associated other classes.
"""

from . import base

import numpy


class EFLPoisRegSimple(base.EFL_GoalModel):
    """Poisson Regression model based on Numberphile video with Tony Padilla
    https://www.numberphile.com/videos/a-million-simulated-seasons
    **But simplified, by assuming equal homefield advantage for all teams.
    """
    
    def __init__(self, eflgames, prior=None, **kwargs):
        """Parameters:
            eflgames - an EFLGames instance
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        team_names = [t.shortname for t in eflgames.teams]
        # Create priors
        if prior is None:
            prior = EFLPoisRegSimple_Prior.default_prior(team_names)
        # Create parameter mapping
        efl2stan = {'HomeGoals':'log_home_goals', 'AwayGoals':'log_away_goals'}
        for i,t in enumerate(team_names):
            efl2stan[t+' Off'] = 'offense[{}]'.format(i+1)
            efl2stan[t+' Def'] = 'defense[{}]'.format(i+1)
        pargroups = {'goals':['HomeGoals','AwayGoals'],
                     'offense':[t+' Off' for t in team_names],
                     'defense':[t+' Def' for t in team_names]}
        for t in team_names:
            pargroups[t] = [t+' Off', t+' Def']
        # Call super init
        super().__init__(
                modelfile      = 'poisreg_simple',
                eflgames       = eflgames,
                extramodeldata = prior.get_params(team_names),
                efl2stan       = efl2stan,
                pargroups      = pargroups,
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


class EFLPoisRegSimple_Prior(object):
    """A class holding a prior for the EFLPoisRegSimple model."""
    
    def __init__(self, offense_prior_mean, offense_prior_var, 
                 defense_prior_mean, defense_prior_var, team_names,
                 log_home_goals_prior_mean, log_home_goals_prior_sd,
                 log_away_goals_prior_mean, log_away_goals_prior_sd):
        """This constructor is pretty much never called in most typical uses.
        Parameters:
            offense_prior_mean - a 1d numpy array containing prior means of
                team offense parameters. Centers it around 0.
            offense_prior_var - a 2d numpy array containing the prior covariance
                of team offense parameters. Assumes it is symmetric pos-def, 
                does no checks.
            defense_prior_mean - a 1d numpy array containing prior means of
                team defense parameters. Centers it around 0.
            defense_prior_var - a 2d numpy array containing the prior covariance
                of team defense parameters. Assumes it is symmetric pos-def, 
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
        self._offense_prior_mean = offense_prior_mean - offense_prior_mean.mean()
        self._defense_prior_mean = defense_prior_mean - defense_prior_mean.mean()
        # Accept the variance matrix as is (future enhancement?)
        self._offense_prior_var = offense_prior_var
        self._defense_prior_var = defense_prior_var
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
        return {'offense_prior_mean':self._offense_prior_mean[idx],
                'offense_prior_var':self._offense_prior_var[idx,:][:,idx],
                'defense_prior_mean':self._defense_prior_mean[idx],
                'defense_prior_var':self._defense_prior_var[idx,:][:,idx],
                'log_home_goals_prior_mean':self._log_home_goals_prior_mean,
                'log_home_goals_prior_sd':self._log_home_goals_prior_sd,
                'log_away_goals_prior_mean':self._log_away_goals_prior_mean,
                'log_away_goals_prior_sd':self._log_away_goals_prior_sd}
        
    # Class methods for creating instances through various methods
    
    @classmethod
    def default_prior(cls, team_names):
        """Instantiates a wide, default prior for this model."""
        P = len(team_names)
        return cls(offense_prior_mean = numpy.zeros(P),
                   offense_prior_var = numpy.identity(P),
                   defense_prior_mean = numpy.zeros(P),
                   defense_prior_var = numpy.identity(P),
                   team_names = team_names,
                   log_home_goals_prior_mean = 0,
                   log_home_goals_prior_sd = 1,
                   log_away_goals_prior_mean = 0,
                   log_away_goals_prior_sd = 1)