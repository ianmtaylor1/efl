#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symordreg.py

Contains the EFLSymOrdReg model and associated other classes.
"""

from . import base

import numpy


##############################################################################
## EFLSymOrdRegHTI ###########################################################
## Symmetric Ordinal Regression Model with Home-Team Interaction #############
##############################################################################


class EFLSymOrdRegHTI(base.EFL_ResultModel):
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
            prior = EFLSymOrdRegHTI_Prior.default_prior(team_names)
        # Create parameter mapping
        efl2stan = {'DrawBoundary':'theta', 'HomeField':'homefield'}
        pargroups = {'other':['DrawBoundary', 'HomeField']}
        for i,t in enumerate(team_names):
            efl2stan[t+' H'] = 'home[{}]'.format(i+1)
            efl2stan[t+' A'] = 'away[{}]'.format(i+1)
            pargroups[t] = [t+' H', t+' A']
        pargroups['home'] = [t+' H' for t in team_names]
        pargroups['away'] = [t+' A' for t in team_names]
        # Call super init
        super().__init__(
                modelfile      = 'symordreg_hti',
                eflgames       = eflgames,
                extramodeldata = prior.get_params(team_names),
                efl2stan       = efl2stan,
                pargroups      = pargroups,
                **kwargs)
    
    def _stan_inits(self, chain_id=None):
        """Draw from a multivariate normal distribution and a logistic
        distribution to produce prior values for beta and theta."""
        P = self._modeldata['home_prior_mean'].shape[0]
        home = numpy.random.multivariate_normal(
                self._modeldata['home_prior_mean'], 
                self._modeldata['home_prior_var'])
        home_raw = (home - home.mean())[:(P-1)]
        away = numpy.random.multivariate_normal(
                self._modeldata['away_prior_mean'], 
                self._modeldata['away_prior_var'])
        away_raw = (away - away.mean())[:(P-1)]
        homefield = numpy.random.normal(
                self._modeldata['homefield_prior_mean'],
                self._modeldata['homefield_prior_sd'])
        theta = abs(numpy.random.logistic(
                self._modeldata['theta_prior_loc'], 
                self._modeldata['theta_prior_scale']))
        return {'home_raw':home_raw, 'away_raw':away_raw, 
                'homefield':homefield, 'theta':theta}


############################################################
## EFLSymOrdRegHTI_Prior ###################################
## Prior for the above model ###############################
############################################################


class EFLSymOrdRegHTI_Prior(object):
    """A class holding a prior for the EFLSymOrdReg model."""
    
    def __init__(self, home_prior_mean, home_prior_var,
                 away_prior_mean, away_prior_var, team_names,
                 homefield_prior_mean, homefield_prior_sd, theta_prior_loc, 
                 theta_prior_scale):
        """This constructor is pretty much never called in most typical uses.
        Parameters:
            home_prior_mean - a 1d numpy array containing prior means of
                team home strength parameters. Centers it around 0.
            home_prior_var - a 2d numpy array containing the prior covariance
                of team away strength parameters. Assumes it is symmetric
                pos-def, does no checks.
            away_prior_mean - a 1d numpy array containing prior means of
                team away strength parameters. Centers it around 0.
            away_prior_var - a 2d numpy array containing the prior covariance
                of team away strength parameters. Assumes it is symmetric
                pos-def, does no checks.
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
        self._home_prior_mean = home_prior_mean - home_prior_mean.mean()
        self._away_prior_mean = away_prior_mean - away_prior_mean.mean()
        # Accept the variance matrix as is (future enhancement?)
        self._home_prior_var = home_prior_var
        self._away_prior_var = away_prior_var
        # Turn the list into a dict of indices
        self._team_map = {t:i for i,t in enumerate(team_names)}
        # Copy all the other parameters
        self._homefield_prior_mean = homefield_prior_mean
        self._homefield_prior_sd = homefield_prior_sd
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
        return {'home_prior_mean':self._home_prior_mean[idx],
                'home_prior_var':self._home_prior_var[idx,:][:,idx],
                'away_prior_mean':self._away_prior_mean[idx],
                'away_prior_var':self._away_prior_var[idx,:][:,idx],
                'homefield_prior_mean':self._homefield_prior_mean,
                'homefield_prior_sd':self._homefield_prior_sd,
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
                 relegated_in=[], promoted_out=[],
                 promoted_in=[], relegated_out=[]):
        """Create a prior from the posterior of a previous EFLSymOrdReg fit.
        Parameters:
            fit - the previous instance of EFLSymOrdReg
            spread - factor by which to inflate variances of all parameters
                from the posterior of 'fit'. Think of this as season-to-season
                uncertainty.
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
        # Get the posterior samples from the fit
        home_df = fit.to_dataframe('home', diagnostics=False)
        away_df = fit.to_dataframe('away', diagnostics=False)
        hdb_df = fit.to_dataframe(['HomeField','DrawBoundary'], diagnostics=False)
        # Determine homefield advantage priors
        homefield_prior_mean = hdb_df['HomeField'].mean()
        homefield_prior_sd = hdb_df['HomeField'].std() * numpy.sqrt(spread)
        # Determine draw boundary priors
        theta_prior_loc = hdb_df['DrawBoundary'].mean()
        theta_prior_scale = hdb_df['DrawBoundary'].std() * 0.5513 * numpy.sqrt(spread)
        # Build parameter names for promoted/relegated teams
        promoted_out_home = [t+' H' for t in promoted_out]
        promoted_out_away = [t+' A' for t in promoted_out]
        relegated_out_home = [t+' H' for t in relegated_out]
        relegated_out_away = [t+' A' for t in relegated_out]
        promoted_in_home = [t+' H' for t in promoted_in]
        promoted_in_away = [t+' A' for t in promoted_in]
        relegated_in_home = [t+' H' for t in relegated_in]
        relegated_in_away = [t+' A' for t in relegated_in]
        # Shuffle and rename the promoted/relegated teams
        for i in home_df.index:
            home_df.loc[i,relegated_out_home] = numpy.random.permutation(home_df.loc[i,relegated_out_home])
            home_df.loc[i,promoted_out_home] = numpy.random.permutation(home_df.loc[i,promoted_out_home])
        for i in away_df.index:
            away_df.loc[i,relegated_out_away] = numpy.random.permutation(away_df.loc[i,relegated_out_away])
            away_df.loc[i,promoted_out_away] = numpy.random.permutation(away_df.loc[i,promoted_out_away])
        home_colmap = {o:i for o,i in zip(relegated_out_home, promoted_in_home)}
        home_colmap.update({o:i for o,i in zip(promoted_out_home, relegated_in_home)})
        away_colmap = {o:i for o,i in zip(relegated_out_away, promoted_in_away)}
        away_colmap.update({o:i for o,i in zip(promoted_out_away, relegated_in_away)})
        home_df = home_df.rename(columns=home_colmap)
        away_df = away_df.rename(columns=away_colmap)
        # Teams priors
        team_names = [c[:-2] for c in home_df.columns if c[-2:]==' H']
        home_names = [t+' H' for t in team_names]
        away_names = [t+' A' for t in team_names]
        home_prior_mean = numpy.array(home_df[home_names].mean()) * regression
        home_prior_var = numpy.cov(home_df[home_names].T)
        away_prior_mean = numpy.array(away_df[away_names].mean()) * regression
        away_prior_var = numpy.cov(away_df[away_names].T)
        # Scale the variance by the spread factor, add small identity for 
        # non-singularity
        num_teams = len(team_names)
        minvar_home = min(home_prior_var.diagonal())
        home_prior_var = (home_prior_var * spread + 
                           numpy.identity(num_teams) * minvar_home * 0.01)
        minvar_away = min(away_prior_var.diagonal())
        away_prior_var = (away_prior_var * spread + 
                           numpy.identity(num_teams) * minvar_away * 0.01)
        # Assemble and return
        return cls(home_prior_mean = home_prior_mean,
                   home_prior_var = home_prior_var,
                   away_prior_mean = away_prior_mean,
                   away_prior_var = away_prior_var,
                   team_names = team_names,
                   homefield_prior_mean = homefield_prior_mean,
                   homefield_prior_sd = homefield_prior_sd,
                   theta_prior_loc = theta_prior_loc,
                   theta_prior_scale = theta_prior_scale)
