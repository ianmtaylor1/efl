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
    

class EFLSymOrdReg_Prior(object):
    """A class holding a prior for the EFLSymOrdReg model."""
    
    def __init__(self, teams_prior_mean, teams_prior_var, team_names,
                 home_prior_mean, home_prior_sd, theta_prior_loc, 
                 theta_prior_scale):
        """This constructor is pretty much never called in most typical uses.
        Parameters:
            teams_prior_mean - a 1d numpy array containing prior means of
                team parameters. Centers it around 0.
            teams_prior_var - a 2d numpy array containing the prior covariance
                of team parameters. Assumes it is symmetric pos-def, does no
                checks.
            team_names - a list of names used as labels for the indices of
                each of the other two parameters. i.e. the names of the 
                elements in teams_prior_mean and the rows and columns of
                teams_prior_var should be in the order of this list.
            home_prior_mean, home_prior_sd - prior parameters for the home
                field advantage parameter
            theta_prior_loc, theta_prior_scale - prior parameters for the 
                draw boundary (theta) parameter
        """
        # Center mean around zero (due to the model's parameter centering)
        self._teams_prior_mean = teams_prior_mean - teams_prior_mean.mean()
        # Accept the variance matrix as is (future enhancement?)
        self._teams_prior_var = teams_prior_var
        # Turn the list into a dict of indices
        self._team_map = {t:i for i,t in enumerate(team_names)}
        # Copy all the other parameters
        self._home_prior_mean = home_prior_mean
        self._home_prior_sd = home_prior_sd
        self._theta_prior_loc = theta_prior_loc
        self._theta_prior_scale = theta_prior_scale
        
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
        return {'teams_prior_mean':self._teams_prior_mean[idx],
                'teams_prior_var':self._teams_prior_var[idx,:][:,idx],
                'home_prior_mean':self._home_prior_mean,
                'home_prior_sd':self._home_prior_sd,
                'theta_prior_loc':self._theta_prior_loc,
                'theta_prior_scale':self._theta_prior_scale}
    
    # Class methods for creating instances through various methods
    
    @classmethod
    def default_prior(cls, team_names):
        """Instantiates a wide, default prior for this model."""
        P = len(team_names)
        return cls(teams_prior_mean = numpy.zeros(P),
                   teams_prior_var = numpy.identity(P) * ((P/2)**2),
                   team_names = team_names,
                   home_prior_mean = 0, home_prior_sd = 1.8138, 
                   theta_prior_loc = 0, theta_prior_scale = 1)
    
    @classmethod
    def from_fit(cls, fit, spread_factor=1, 
                 relegated_in=[], promoted_out=[],
                 promoted_in=[], relegated_out=[]):
        """Create a prior from the posterior of a previous EFLSymOrdReg fit.
        Parameters:
            fit - the previous instance of EFLSymOrdReg
            spread_factor - factor by which to inflate variances of all 
                parameters from the posterior of 'fit'. Think of this as
                season-to-season uncertainty.
            relegated_in - list of team names who were relegated into this
                league from a higher league between the season of 'fit' and now
            promoted_out - list of team names who were promoted out of this
                league into a higher league between the season of 'fit' and now
            promoted_in - list of team names who were promoted into this
                league from a lower league between the season of 'fit' and now
            relegated_out - list of team names who were relegated out of this
                league into a lower league between the season of 'fit' and now
        """
        raise NotImplementedError('from_fit not implemented')