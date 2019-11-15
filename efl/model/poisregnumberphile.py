#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poisregnumberphile.py

Contains the EFLPoisRegNumberphile model and associated other classes.
"""

from . import base

import numpy


class EFLPoisRegNumberphile(base.EFL_GoalModel):
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
        pargroups = {'goals':['HomeGoals','AwayGoals'],
                     'homeoff':[t.shortname+' HO' for t in eflgames.teams],
                     'homedef':[t.shortname+' HD' for t in eflgames.teams],
                     'awayoff':[t.shortname+' AO' for t in eflgames.teams],
                     'awaydef':[t.shortname+' AD' for t in eflgames.teams]}
        for t in eflgames.teams:
            pargroups[t.shortname] = [t.shortname+' HO', t.shortname+' HD',
                                      t.shortname+' AO', t.shortname+' AD']
        # Call super init
        super().__init__(
                modelfile      = 'poisreg_numberphile',
                eflgames       = eflgames,
                extramodeldata = priors,
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
    
