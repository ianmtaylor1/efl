#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compoisregefgm.py

Contains the COMPoisRegEFGM model and associated other classes.
"""

from . import base
from .. import util

import numpy


class COMPoisRegEFGM(base.GoalModel):
    """Generalized Poisson Regression model using the Conway-Maxwell-Poisson
    distribution for goals. 
    https://en.wikipedia.org/wiki/Conway%E2%80%93Maxwell%E2%80%93Poisson_distribution
    No separate homefield advantage effects (more similar to PoisRegSimple
    than PoisRegNumberphile)
    """
    
    def __init__(self, eflgames, prior=None,
                 nu_lower_limit=0.0, truncpoint=5000, **kwargs):
        """Parameters:
            eflgames - a Games instance
            prior - instance of COMPoisReg_Prior, or None for a diffuse 
                prior (default)
            nu_lower_limit - lower bound to place on concentration parameter,
                nu. A higher value limits overdispersion in the model. Minimum
                of zero.
            truncpoint - hard truncation point to apply to the COM-Poisson
                distribution. Should be at least 100*(max + 1), where 'max' is
                the maximum number of goals scored by any team in a match.
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        team_names = [t.shortname for t in eflgames.teams]
        # Create priors
        if prior is None:
            prior = COMPoisRegEFGM_Prior.default_prior(team_names)
        # Create parameter mapping
        efl2stan = {'HomeGoals':'log_home_goals', 'AwayGoals':'log_away_goals',
                    'GoalsConcentration':'nu', 'GoalsCorr':'rho'}
        for i,t in enumerate(team_names):
            efl2stan[t+' Off'] = 'offense[{}]'.format(i+1)
            efl2stan[t+' Def'] = 'defense[{}]'.format(i+1)
        pargroups = {'goals':['HomeGoals','AwayGoals','GoalsConcentration',
                              'GoalsCorr'],
                     'offense':[t+' Off' for t in team_names],
                     'defense':[t+' Def' for t in team_names]}
        for t in team_names:
            pargroups[t] = [t+' Off', t+' Def']
        # Call super init
        super().__init__(
                modelfile      = 'compoisreg_efgm',
                eflgames       = eflgames,
                extramodeldata = {'nu_lower_limit':nu_lower_limit,
                                  'truncpoint':truncpoint, 
                                  **prior.get_params(team_names)},
                efl2stan       = efl2stan,
                pargroups      = pargroups,
                **kwargs)


class COMPoisRegEFGM_Prior(object):
    """A class holding a prior for the COMPoisReg model."""
    
    def __init__(self, offense_prior_mean, offense_prior_var, 
                 defense_prior_mean, defense_prior_var, team_names,
                 log_home_goals_prior_mean, log_home_goals_prior_sd,
                 log_away_goals_prior_mean, log_away_goals_prior_sd,
                 nu_prior_mean, nu_prior_sd, phi_prior_mean, phi_prior_sd):
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
            nu_prior_mean, nu_prior_sd - prior parameters for
                the goals dispersion/decay parameter.
            phi_prior_mean, phi_prior_sd - prior parameters for the EFGM
                correlation parameter.
        """
        # Center mean around zero (due to the model's parameter centering)
        self._offense_prior_mean = offense_prior_mean - offense_prior_mean.mean()
        self._defense_prior_mean = defense_prior_mean - defense_prior_mean.mean()
        # Accept the variance matrix as is (future enhancement?)
        self._offense_prior_var = offense_prior_var
        self._defense_prior_var = defense_prior_var
        # Turn the list into a dict of indices
        self._team_map = {t:i for i,t in enumerate(team_names)}
        # Copy all the log baseline goals parameters
        self._log_home_goals_prior_mean = log_home_goals_prior_mean
        self._log_home_goals_prior_sd = log_home_goals_prior_sd
        self._log_away_goals_prior_mean = log_away_goals_prior_mean
        self._log_away_goals_prior_sd = log_away_goals_prior_sd
        # Copy the dispersion parameters
        self._nu_prior_mean = nu_prior_mean
        self._nu_prior_sd = nu_prior_sd
        # Copy the correlation parameters
        self._phi_prior_mean = phi_prior_mean
        self._phi_prior_sd = phi_prior_sd
        
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
                'log_away_goals_prior_sd':self._log_away_goals_prior_sd,
                'nu_prior_mean':self._nu_prior_mean,
                'nu_prior_sd':self._nu_prior_sd,
                'phi_prior_mean':self._phi_prior_mean,
                'phi_prior_sd':self._phi_prior_sd}
        
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
                   log_away_goals_prior_sd = 1,
                   nu_prior_mean = 1,
                   nu_prior_sd = 0.25,
                   phi_prior_mean = 0,
                   phi_prior_sd = 1)
        
    @classmethod
    def from_fit(cls, fit, spread=1.0, regression=1.0,
                 relegated_in=[], promoted_out=[],
                 promoted_in=[], relegated_out=[]):
        """Create a prior from the posterior of a previous COMPoisReg fit.
        Parameters:
            fit - the previous instance of ConsulJainReg
            spread - factor by which to inflate variances of all parameters
                from the posterior of 'fit'. Think of this as season-to-season
                uncertainty.
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
        # Get the posterior samples from the fit
        df = fit.to_dataframe(diagnostics=False)
        # Goals parameters
        log_home_goals_prior_mean = df['HomeGoals'].mean()
        log_home_goals_prior_sd = df['HomeGoals'].std() * numpy.sqrt(spread)
        log_away_goals_prior_mean = df['AwayGoals'].mean()
        log_away_goals_prior_sd = df['AwayGoals'].std() * numpy.sqrt(spread)
        # Dispersion parameter
        nu_prior_mean = df['GoalsConcentration'].mean()
        nu_prior_sd = df['GoalsConcentration'].std() * numpy.sqrt(spread)
        # Correlation parameter
        phi_prior_mean = df['GoalsCorr'].mean() * 3
        phi_prior_sd = df['GoalsCorr'].std() * numpy.sqrt(spread) * 3
        # Relegation/Promotion
        if len(promoted_out) > 0:
            util.shuffle_rename(df, [t+' Off' for t in promoted_out],
                                newnames=[t+' Off' for t in relegated_in])
            util.shuffle_rename(df, [t+' Def' for t in promoted_out],
                                newnames=[t+' Def' for t in relegated_in])
        if len(relegated_out) > 0:
            util.shuffle_rename(df, [t+' Off' for t in relegated_out],
                                newnames=[t+' Off' for t in promoted_in])
            util.shuffle_rename(df, [t+' Def' for t in relegated_out],
                                newnames=[t+' Def' for t in promoted_in])
        # Team offense/defense priors
        team_names = [c[:-4] for c in df.columns if c[-4:]==' Off']
        offense_prior_mean, offense_prior_var = util.mean_var(
                df, cols=[t+' Off' for t in team_names],
                meanregress=regression, varspread=spread, ridge=0.0001)
        defense_prior_mean, defense_prior_var = util.mean_var(
                df, cols=[t+' Def' for t in team_names],
                meanregress=regression, varspread=spread, ridge=0.0001)
        # Assemble and return
        return cls(offense_prior_mean = offense_prior_mean,
                   offense_prior_var = offense_prior_var,
                   defense_prior_mean = defense_prior_mean,
                   defense_prior_var = defense_prior_var,
                   team_names = team_names,
                   log_home_goals_prior_mean = log_home_goals_prior_mean,
                   log_home_goals_prior_sd = log_home_goals_prior_sd,
                   log_away_goals_prior_mean = log_away_goals_prior_mean,
                   log_away_goals_prior_sd = log_away_goals_prior_sd,
                   nu_prior_mean = nu_prior_mean,
                   nu_prior_sd = nu_prior_sd,
                   phi_prior_mean = phi_prior_mean,
                   phi_prior_sd = phi_prior_sd)

