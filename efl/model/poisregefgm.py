#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poisregefgm.py

Contains the PoisRegEFGM model and associated other classes.
"""

from . import base

import numpy


class PoisRegEFGM(base.GoalModel):
    """Poisson Regression model with an EFGM copula between homegoals and
    awaygoals to allow correlation.
    """
    
    def __init__(self, eflgames, prior=None, **kwargs):
        """Parameters:
            eflgames - an EFLGames instance
            prior - instance of PoisRegEFGM_Prior, or None for a diffuse 
                prior (default)
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        team_names = [t.shortname for t in eflgames.teams]
        # Create priors
        if prior is None:
            prior = PoisRegEFGM_Prior.default_prior(team_names)
        # Create parameter mapping
        efl2stan = {'HomeGoals':'log_home_goals', 'AwayGoals':'log_away_goals',
                    'GoalsCorr':'rho'}
        for i,t in enumerate(team_names):
            efl2stan[t+' Off'] = 'offense[{}]'.format(i+1)
            efl2stan[t+' Def'] = 'defense[{}]'.format(i+1)
        pargroups = {'goals':['HomeGoals','AwayGoals','GoalsCorr'],
                     'offense':[t+' Off' for t in team_names],
                     'defense':[t+' Def' for t in team_names]}
        for t in team_names:
            pargroups[t] = [t+' Off', t+' Def']
        # Call super init
        super().__init__(
                modelfile      = 'poisreg_efgm',
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
        rho = 0.0
        return {'log_home_goals':log_home_goals, 
                'log_away_goals':log_away_goals,
                'offense_raw':offense_raw,
                'defense_raw':defense_raw,
                'rho':rho}


class PoisRegEFGM_Prior(object):
    """A class holding a prior for the EFLPoisRegEFGM model."""
    
    def __init__(self, offense_prior_mean, offense_prior_var, 
                 defense_prior_mean, defense_prior_var, team_names,
                 log_home_goals_prior_mean, log_home_goals_prior_sd,
                 log_away_goals_prior_mean, log_away_goals_prior_sd,
                 rho_prior_mean, rho_prior_sd):
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
            rho_prior_mean, rho_prior_sd - prior parameters for the EFGM
                copula correlation parameter (scaled by 1/3 to better represent
                Pearson correlation).
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
        self._rho_prior_mean = rho_prior_mean
        self._rho_prior_sd= rho_prior_sd
        
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
                'rho_prior_mean':self._rho_prior_mean,
                'rho_prior_sd':self._rho_prior_sd}
        
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
                   rho_prior_mean = 0,
                   rho_prior_sd = 1)
        
    @classmethod
    def from_fit(cls, fit, spread=1.0, regression=1.0,
                 relegated_in=[], promoted_out=[],
                 promoted_in=[], relegated_out=[]):
        """Create a prior from the posterior of a previous PoisRegSimple fit.
        Parameters:
            fit - the previous instance of PoisRegSimple
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
        # EFGM parameters
        rho_prior_mean = df['GoalsCorr'].mean()
        rho_prior_sd = df['GoalsCorr'].std() * numpy.sqrt(spread)
        # Build parameter names for promoted/relegated teams
        promoted_out_off = [t+' Off' for t in promoted_out]
        promoted_out_def = [t+' Def' for t in promoted_out]
        relegated_out_off = [t+' Off' for t in relegated_out]
        relegated_out_def = [t+' Def' for t in relegated_out]
        promoted_in_off = [t+' Off' for t in promoted_in]
        promoted_in_def = [t+' Def' for t in promoted_in]
        relegated_in_off = [t+' Off' for t in relegated_in]
        relegated_in_def = [t+' Def' for t in relegated_in]
        # Shuffle promoted/relegated teams
        for i in df.index:
            df.loc[i,promoted_out_off] = numpy.random.permutation(df.loc[i,promoted_out_off])
            df.loc[i,promoted_out_def] = numpy.random.permutation(df.loc[i,promoted_out_def])
            df.loc[i,relegated_out_off] = numpy.random.permutation(df.loc[i,relegated_out_off])
            df.loc[i,relegated_out_def] = numpy.random.permutation(df.loc[i,relegated_out_def])
        colmap = {o:i for o,i in zip(promoted_out_off, relegated_in_off)}
        colmap.update({o:i for o,i in zip(promoted_out_def, relegated_in_def)})
        colmap.update({o:i for o,i in zip(relegated_out_off, promoted_in_off)})
        colmap.update({o:i for o,i in zip(relegated_out_def, promoted_in_def)})
        df = df.rename(columns=colmap)
        # Team offense/defense priors
        team_names = [c[:-4] for c in df.columns if c[-4:]==' Off']
        offense_pars = [t+' Off' for t in team_names]
        defense_pars = [t+' Def' for t in team_names]
        offense_prior_mean = numpy.array(df[offense_pars].mean()) * regression
        offense_prior_var = numpy.cov(df[offense_pars].T)
        defense_prior_mean = numpy.array(df[defense_pars].mean()) * regression
        defense_prior_var = numpy.cov(df[defense_pars].T)
        # Scale the variance by the spread factor, add small identity for 
        # non-singularity
        num_teams = len(team_names)
        minvar_off = min(offense_prior_var.diagonal())
        minvar_def = min(defense_prior_var.diagonal())
        offense_prior_var = (offense_prior_var * spread + 
                             numpy.identity(num_teams) * minvar_off * 0.01)
        defense_prior_var = (defense_prior_var * spread + 
                             numpy.identity(num_teams) * minvar_def * 0.01)
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
                   rho_prior_mean = rho_prior_mean,
                   rho_prior_sd = rho_prior_sd)

