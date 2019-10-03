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

