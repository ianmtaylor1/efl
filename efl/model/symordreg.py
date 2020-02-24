#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symordreg.py

Contains the SymOrdReg model and associated other classes.
"""

from . import base
from .. import util

import numpy


############################################################
## SymOrdReg ###############################################
## Simplest Symmetric Ordinal Regression Model #############
############################################################


class SymOrdReg(base.ResultModel):
    """*Sym*metric *Ord*inal *Reg*ression model for EFL data."""
    
    def __init__(self, eflgames, prior=None, **kwargs):
        """Parameters:
            eflgames - a Games instance
            prior - an SymOrdReg_Prior instance, or None for diffuse priors
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        team_names = [t.shortname for t in eflgames.teams]
        # Create priors
        if prior is None:
            prior = SymOrdReg_Prior.default_prior(team_names)
        # Create parameter mapping
        efl2stan = {'DrawBoundary':'theta', 'HomeField':'home'}
        for i,t in enumerate(team_names):
            efl2stan[t] = 'teams[{}]'.format(i+1)
        pargroups = {'other':['DrawBoundary', 'HomeField'], 'teams':team_names}
        # Call super init
        super().__init__(
                modelfile      = 'symordreg',
                eflgames       = eflgames,
                extramodeldata = prior.get_params(team_names),
                efl2stan       = efl2stan,
                pargroups      = pargroups,
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


############################################################
## SymOrdReg_Prior #########################################
## Prior for either of the above models ####################
############################################################


class SymOrdReg_Prior(object):
    """A class holding a prior for the SymOrdReg model."""
    
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
                   teams_prior_var = numpy.identity(P) * ((P/4)**2),
                   team_names = team_names,
                   home_prior_mean = 0, home_prior_sd = 1.8138, 
                   theta_prior_loc = 0, theta_prior_scale = 1)
    
    @classmethod
    def from_fit(cls, fit, spread=1.0, regression=1.0,
                 team_spread=None, other_spread=None,
                 relegated_in=[], promoted_out=[],
                 promoted_in=[], relegated_out=[]):
        """Create a prior from the posterior of a previous SymOrdReg fit.
        Parameters:
            fit - the previous instance of SymOrdReg
            spread - factor by which to inflate variances of all parameters
                from the posterior of 'fit'. Think of this as season-to-season
                uncertainty.
            team_spread, other_spread - variance inflation factors for team
                strength (offense, defense) and other parameters, respectively.
                If supplied, overrides default from spread.
            regression - is multiplied by team means
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
        df = fit.to_dataframe('teams', diagnostics=False)
        hdb_df = fit.to_dataframe(['HomeField','DrawBoundary'], diagnostics=False)
        # Determine homefield advantage priors
        home_prior_mean = hdb_df['HomeField'].mean()
        home_prior_sd = hdb_df['HomeField'].std() * numpy.sqrt(other_spread)
        # Determine draw boundary priors
        theta_prior_loc = hdb_df['DrawBoundary'].mean()
        theta_prior_scale = hdb_df['DrawBoundary'].std() * 0.5513 * numpy.sqrt(other_spread)
        # Promotion/relegation
        if len(promoted_out) > 0:
            util.shuffle_rename(df, promoted_out, newnames=relegated_in)
        if len(relegated_out) > 0:
            util.shuffle_rename(df, relegated_out, newnames=promoted_in)
        # Teams priors
        team_names = list(set(df.columns) - set(['chain','draw','warmup']))
        teams_prior_mean, teams_prior_var = util.mean_var(
                df, cols=team_names, meanregress=regression,
                varspread=team_spread, ridge=0.0001)
        # Assemble and return
        return cls(teams_prior_mean = teams_prior_mean,
                   teams_prior_var = teams_prior_var,
                   team_names = team_names,
                   home_prior_mean = home_prior_mean,
                   home_prior_sd = home_prior_sd,
                   theta_prior_loc = theta_prior_loc,
                   theta_prior_scale = theta_prior_scale)
