#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symordreg.py

Contains the SymOrdReg model and associated other classes.
"""

from .. import base

import numpy


############################################################
## EFLSymOrdMulReg #########################################
## Symmetric Ordinal Regression with Multiplicative Effects
############################################################


class SymOrdMulReg(base.ResultModel):
    """*Sym*metric *Ord*inal *Mul*tiplicative *Reg*ression model for EFL data."""
    
    def __init__(self, eflgames, prior=None, muldim=2, **kwargs):
        """Parameters:
            eflgames - a Games instance
            prior - an SymOrdMulReg_Prior instance, or None for diffuse priors
            muldim - the dimension of the multiplicative effects
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        team_names = [t.shortname for t in eflgames.teams]
        # Create priors
        if prior is None:
            prior = SymOrdMulReg_Prior.default_prior(team_names)
        # Create parameter mapping
        efl2stan = {'DrawBoundary':'theta', 'HomeField':'home'}
        for i,t in enumerate(team_names):
            efl2stan[t] = 'teams[{}]'.format(i+1)
        pargroups = {'teams':team_names}
        matchups = []
        for i1,t1 in enumerate(team_names):
            teampars = []
            for i2,t2 in enumerate(team_names):
                if i1 != i2:
                    parname = '{} vs {}'.format(t1,t2)
                    efl2stan[parname] = 'matchup[{},{}]'.format(i1+1,i2+1)
                    teampars.append(parname)
                    matchups.append(parname)
            pargroups['{} vs'.format(t1)] = teampars
        pargroups['matchups'] = matchups
        scales = []
        for i,t in enumerate(team_names):
            parname = 'matchup_scale[{}]'.format(t)
            efl2stan[parname] = 'matchup_scale[{}]'.format(i+1)
            scales.append(parname)
        pargroups['matchup_scale'] = scales
        # Call super init
        super().__init__(
                modelfile      = 'symordreg_mult',
                eflgames       = eflgames,
                extramodeldata = {'uvdim':muldim, 
                                  **prior.get_params(team_names)},
                efl2stan       = efl2stan,
                pargroups      = pargroups,
                **kwargs)
    
    def _stan_inits(self, chain_id=None):
        """Draw from a multivariate normal distribution and a logistic
        distribution to produce prior values for beta and theta."""
        P = self._modeldata['teams_prior_mean'].shape[0]
        K = self._modeldata['uvdim']
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
        uvscale = numpy.array(sorted(abs(numpy.random.normal(size=K))))
        uvcorr = numpy.identity(2*K)
        UV = numpy.random.multivariate_normal(
                numpy.zeros(2*K),
                numpy.diag(numpy.append(uvscale,uvscale)),
                size=P)
        return {'teams_raw':teams_raw, 'home':home, 'theta':theta,
                'uvscale':uvscale, 'uvcorr_chol':uvcorr,
                'U':UV[:,:K], 'V':UV[:,K:]}


############################################################
## SymOrdMulReg_Prior ######################################
## Prior for either of the above models ####################
############################################################


class SymOrdMulReg_Prior(object):
    """A class holding a prior for the EFLSymOrdReg model."""
    pass