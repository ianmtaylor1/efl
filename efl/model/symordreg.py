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
    
    def __init__(self, eflgames, prior=None, **kwargs):
        """Parameters:
            eflgames - an EFLGames instance
            prior - an EFLSymOrdReg_Prior instance, or None for diffuse priors
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        team_names = [t.shortname for t in eflgames.teams]
        # Create priors
        if prior is None:
            prior = EFLSymOrdReg_Prior.default_prior(team_names)
        # Create parameter mapping
        efl2stan = {'DrawBoundary':'theta', 'HomeField':'home'}
        for i,t in enumerate(eflgames.teams):
            efl2stan[t.shortname] = 'teams[{}]'.format(i+1)
        # Call super init
        super().__init__(
                modelfile      = 'symordreg_v2',
                eflgames       = eflgames,
                extramodeldata = prior.get_params(team_names),
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
        # Get the posterior samples from the fit
        df = fit.to_dataframe(diagnostics=False)
        # Determine homefield advantage priors
        home_prior_mean = df['HomeField'].mean()
        home_prior_sd = df['HomeField'].std() * numpy.sqrt(spread_factor)
        # Determine draw boundary priors
        theta_prior_loc = df['DrawBoundary'].mean()
        theta_prior_scale = df['DrawBoundary'].std() * 0.5513 * numpy.sqrt(spread_factor)
        # Teams priors - base
        team_names_base = list(
                set(df.columns) 
                - set(['DrawBoundary','HomeField','chain','draw','warmup'])
                )
        teams_mean_base = numpy.array(df[team_names_base].mean())
        teams_var_base = numpy.cov(df[team_names_base].T)
        # Do promotion and relegation
        promoted_idx = [team_names_base.index(t) for t in promoted_out]
        relegated_idx = [team_names_base.index(t) for t in relegated_out]
        # Names
        team_names = team_names_base.copy()
        for i in range(len(relegated_in)):
            team_names[promoted_idx[i]] = relegated_in[i]
        for i in range(len(promoted_in)):
            team_names[relegated_idx[i]] = promoted_in[i]
        # Means
        teams_prior_mean = teams_mean_base.copy()
        if len(promoted_idx) > 0:
            teams_prior_mean[promoted_idx] = teams_prior_mean[promoted_idx].mean()
        if len(relegated_idx) > 0:
            teams_prior_mean[relegated_idx] = teams_prior_mean[relegated_idx].mean()
        # Variance: see mixture distrubiton documentation
        teams_prior_var = teams_var_base.copy()
        # Covariance between promoted/relegated teams and all other teams
        for i in promoted_idx:
            teams_prior_var[:,i] = teams_var_base[:,promoted_idx].mean(axis=1)
            teams_prior_var[i,:] = teams_var_base[promoted_idx,:].mean(axis=0)
        for i in relegated_idx:
            teams_prior_var[:,i] = teams_var_base[:,relegated_idx].mean(axis=1)
            teams_prior_var[i,:] = teams_var_base[relegated_idx,:].mean(axis=0)
        # Covariance within promoted/relegated teams
        if len(promoted_idx) > 0:
            promoted_avg = teams_var_base[:,promoted_idx][promoted_idx,:].mean()
        for i in promoted_idx:
            for j in promoted_idx:
                teams_prior_var[i,j] = promoted_avg
        if len(relegated_idx) > 0:
            relegated_avg = teams_var_base[:,relegated_idx][relegated_idx,:].mean()
        for i in relegated_idx:
            for j in relegated_idx:
                teams_prior_var[i,j] = relegated_avg
        # Variance of promoted/relegated teams themselves
        if len(promoted_idx) > 0:
            teams_prior_var[promoted_idx, promoted_idx] = (
                    (teams_prior_var[promoted_idx, promoted_idx]
                    + teams_mean_base[promoted_idx]**2).mean()
                    - teams_mean_base[promoted_idx].mean()**2)
        if len(relegated_idx) > 0:
            teams_prior_var[relegated_idx, relegated_idx] = (
                    (teams_prior_var[relegated_idx, relegated_idx]
                    + teams_mean_base[relegated_idx]**2).mean()
                    - teams_mean_base[relegated_idx].mean()**2)
        # Scale the variance by the spread factor, add small identity for 
        # non-singularity
        teams_prior_var *= spread_factor
        numpy.fill_diagonal(teams_prior_var, teams_prior_var.diagonal() * 1.01)
        # Assemble and return
        return cls(teams_prior_mean = teams_prior_mean,
                   teams_prior_var = teams_prior_var,
                   team_names = team_names,
                   home_prior_mean = home_prior_mean,
                   home_prior_sd = home_prior_sd,
                   theta_prior_loc = theta_prior_loc,
                   theta_prior_scale = theta_prior_scale)
