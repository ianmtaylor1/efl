#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symordreg.py

Contains the EFLSymOrdReg model and associated other classes.
"""

from . import base

import numpy


class EFLSymOrdReg(base.EFL_ResultModel):
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