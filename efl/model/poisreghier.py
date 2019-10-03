#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poisregsimple.py

Contains the EFLPoisRegSimple model and associated other classes.
"""

from . import base

import numpy


class EFLPoisRegHier(base.EFL_GoalModel):
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

  