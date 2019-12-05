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
    
    def __init__(self, eflgames, prior = None, **kwargs):
        """Parameters:
            eflgames - an EFLGames instance
            prior - an instance of EFLPoisRegHier_Prior, or None for a diffuse
                prior (default)
            **kwargs - extra arguments passed to base models (usually Stan
                sampling options)
        """
        # Create priors
        team_names = [t.shortname for t in eflgames.teams]
        # Create priors
        if prior is None:
            prior = EFLPoisRegHier_Prior.default_prior(team_names)
        # Create parameter mapping
        efl2stan = {'LogAvgGoals':'log_goals', 'HomeField':'home',
                    'TeamSkillSD':'sigma',
                    'Cor(HO,AO)':'teamcorr[1,2]', 'Cor(HO,HD)':'teamcorr[1,3]',
                    'Cor(HO,AD)':'teamcorr[1,4]', 'Cor(AO,HD)':'teamcorr[2,3]',
                    'Cor(AO,AD)':'teamcorr[2,4]', 'Cor(HD,AD)':'teamcorr[3,4]'}
        for i,t in enumerate(eflgames.teams):
            efl2stan[t.shortname] = 'teams[{}]'.format(i+1)
            efl2stan[t.shortname+' HO'] = 'homeoff[{}]'.format(i+1)
            efl2stan[t.shortname+' HD'] = 'homedef[{}]'.format(i+1)
            efl2stan[t.shortname+' AO'] = 'awayoff[{}]'.format(i+1)
            efl2stan[t.shortname+' AD'] = 'awaydef[{}]'.format(i+1)
        pargroups = {'teams':team_names,
                     'teamvar':['TeamSkillSD', 'Cor(HO,AO)', 'Cor(HO,HD)',
                                'Cor(HO,AD)', 'Cor(AO,HD)', 'Cor(AO,AD)',
                                'Cor(HD,AD)'],
                     'homeoff':[t+' HO' for t in team_names],
                     'homedef':[t+' HD' for t in team_names],
                     'awayoff':[t+' AO' for t in team_names],
                     'awaydef':[t+' AD' for t in team_names]}
        for t in team_names:
            pargroups[t+' All'] = [t, t+' HO', t+' HD', t+' AO', t+' AD']
        # Call super init
        super().__init__(
                modelfile      = 'poisreg_hier',
                eflgames       = eflgames,
                extramodeldata = prior.get_params(team_names),
                efl2stan       = efl2stan,
                pargroups      = pargroups,
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
        if self._modeldata['sigma_prior_informative']:
            sigma = numpy.sqrt(1.0 / numpy.random.gamma(
                    self._modeldata['sigma_prior_alpha'],
                    1/self._modeldata['sigma_prior_beta']))
        else:
            # Don't actually sample from Cauchy for starting values
            sigma = abs(numpy.random.standard_normal())
        # TODO: figure out how to sample LKJ in numpy
        teamcorr = numpy.identity(4)
        # Individual team strengths
        teams = numpy.random.multivariate_normal(
                self._modeldata['teams_prior_mean'],
                self._modeldata['teams_prior_var'])
        teams = teams - teams.mean()
        # Team subparameters
        P = self._modeldata['teams_prior_mean'].shape[0]
        homeoff = numpy.zeros(P)
        homedef = numpy.zeros(P)
        awayoff = numpy.zeros(P)
        awaydef = numpy.zeros(P)
        for i in range(P):
            mean = numpy.array([home, 0, home, 0]) + teams[i]
            x = numpy.random.multivariate_normal(mean, teamcorr * sigma**2)
            homeoff[i] = x[0]
            awayoff[i] = x[1]
            homedef[i] = x[2]
            awaydef[i] = x[3]
        # Put together and return
        return {'log_goals':log_goals, 'home':home, 'teams_raw':teams[:(P-1)],
                'sigma':sigma, 'teamcorr_chol':numpy.linalg.cholesky(teamcorr),
                'homeoff':homeoff, 'homedef':homedef,
                'awayoff':awayoff, 'awaydef':awaydef}


class EFLPoisRegHier_Prior(object):
    """A class holding a prior for the EFLPoisRegHier model."""
    
    def __init__(self, team_names, 
                 teams_prior_mean, teams_prior_var, 
                 log_goals_prior_mean, log_goals_prior_sd,
                 home_prior_mean, home_prior_sd,
                 sigma_prior_informative,
                 sigma_prior_alpha, sigma_prior_beta,
                 teamcorr_prior_eta):
        """This constructor is pretty much never called in most typical uses.
        Parameters:
            team_names - a list of names used as labels for the indices of
                each of the team parameters. i.e. the names of the elements
                in teams_prior_mean and the rows and columns of 
                teams_prior_var should be in the order of this list.
            teams_prior_mean - a 1d numpy array containing prior means of
                team parameters. Centers it around 0.
            teams_prior_var - a 2d numpy array containing the prior covariance
                of team parameters. Assumes it is symmetric pos-def, does no
                checks.
            log_goals_prior_mean, log_goals_prior_sd - prior parameters for
                the goals scored parameter
            home_prior_mean, home_prior_sd - prior parameters for the homefield
                advantage parameter
            sigma_prior_informative - (bool) whether to use an informative
                (inverse-gamma) prior on sigma^2, rather than a diffuse 
                (cauchy) prior on sigma.
            sigma_prior_alpha, sigma_prior_beta - if using an informative
                sigma prior, the alpha and beta inverse-gamma parameters
            teamcorr_prior_eta - the LKJ parameter for the correlation matrix
                of hierarchical team parameters
        """
        # Center mean around zero (due to the model's parameter centering)
        self._teams_prior_mean = teams_prior_mean - teams_prior_mean.mean()
        # Accept the variance matrix as is (future enhancement?)
        self._teams_prior_var = teams_prior_var
        # Turn the list into a dict of indices
        self._team_map = {t:i for i,t in enumerate(team_names)}
        # Copy all the other parameters
        self._log_goals_prior_mean = log_goals_prior_mean
        self._log_goals_prior_sd = log_goals_prior_sd
        self._home_prior_mean = home_prior_mean
        self._home_prior_sd = home_prior_sd
        self._sigma_prior_informative = 1 if sigma_prior_informative else 0
        self._sigma_prior_alpha = sigma_prior_alpha
        self._sigma_prior_beta = sigma_prior_beta
        self._teamcorr_prior_eta = teamcorr_prior_eta
        
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
                'log_goals_prior_mean':self._log_goals_prior_mean,
                'log_goals_prior_sd':self._log_goals_prior_sd,
                'home_prior_mean':self._home_prior_mean,
                'home_prior_sd':self._home_prior_sd,
                'sigma_prior_informative':self._sigma_prior_informative,
                'sigma_prior_alpha':self._sigma_prior_alpha,
                'sigma_prior_beta':self._sigma_prior_beta,
                'teamcorr_prior_eta':self._teamcorr_prior_eta}
        
    # Class methods for creating instances through various methods
    
    @classmethod
    def default_prior(cls, team_names):
        """Instantiates a wide, default prior for this model."""
        P = len(team_names)
        return cls(team_names = team_names,
                   teams_prior_mean = numpy.zeros(P),
                   teams_prior_var = numpy.identity(P),
                   log_goals_prior_mean = 0,
                   log_goals_prior_sd = 1,
                   home_prior_mean = 0,
                   home_prior_sd = 1,
                   sigma_prior_informative = False,
                   sigma_prior_alpha = 1,
                   sigma_prior_beta = 1,
                   teamcorr_prior_eta = 1)
        
#    @classmethod
#    def from_fit(cls, fit, spread=1.0, regression=1.0,
#                 relegated_in=[], promoted_out=[],
#                 promoted_in=[], relegated_out=[]):
#        """Create a prior from the posterior of a previous EFLPoisRegNumberphile fit.
#        Parameters:
#            fit - the previous instance of EFLPoisRegNumberphile
#            spread - factor by which to inflate variances of all parameters
#                from the posterior of 'fit'. Think of this as season-to-season
#                uncertainty.
#            regression - is multiplied by team offense/defense means
#            relegated_in - list of team names who were relegated into this
#                league from a higher league between the season of 'fit' and now
#            promoted_out - list of team names who were promoted out of this
#                league into a higher league between the season of 'fit' and now
#            promoted_in - list of team names who were promoted into this
#                league from a lower league between the season of 'fit' and now
#            relegated_out - list of team names who were relegated out of this
#                league into a lower league between the season of 'fit' and now
#        """
#        # Input checks
#        if len(relegated_out) != len(promoted_in):
#            raise Exception('len(relegated_out) must equal len(promoted_in)')
#        if len(promoted_out) != len(relegated_in):
#            raise Exception('len(promoted_out) must equal len(relegated_in)')
#        if len(set(relegated_out) & set(promoted_out)) > 0:
#            raise Exception('promoted_out and relegated_out cannot have common teams')
#        if len(set(relegated_in) & set(promoted_in)) > 0:
#            raise Exception('promoted_in and relegated_in cannot have common teams')
#        # Get the posterior samples from the fit
#        df = fit.to_dataframe(diagnostics=False)
#        # Goals parameters
#        log_home_goals_prior_mean = df['HomeGoals'].mean()
#        log_home_goals_prior_sd = df['HomeGoals'].std() * numpy.sqrt(spread)
#        log_away_goals_prior_mean = df['AwayGoals'].mean()
#        log_away_goals_prior_sd = df['AwayGoals'].std() * numpy.sqrt(spread)
#        # Build parameter names for promoted/relegated teams
#        promoted_out_ho = [t+' HO' for t in promoted_out]
#        promoted_out_hd = [t+' HD' for t in promoted_out]
#        promoted_out_ao = [t+' AO' for t in promoted_out]
#        promoted_out_ad = [t+' AD' for t in promoted_out]
#        relegated_out_ho = [t+' HO' for t in relegated_out]
#        relegated_out_hd = [t+' HD' for t in relegated_out]
#        relegated_out_ao = [t+' AO' for t in relegated_out]
#        relegated_out_ad = [t+' AD' for t in relegated_out]
#        promoted_in_ho = [t+' HO' for t in promoted_in]
#        promoted_in_hd = [t+' HD' for t in promoted_in]
#        promoted_in_ao = [t+' AO' for t in promoted_in]
#        promoted_in_ad = [t+' AD' for t in promoted_in]
#        relegated_in_ho = [t+' HO' for t in relegated_in]
#        relegated_in_hd = [t+' HD' for t in relegated_in]
#        relegated_in_ao = [t+' AO' for t in relegated_in]
#        relegated_in_ad = [t+' AD' for t in relegated_in]
#        # Shuffle promoted/relegated teams
#        for i in df.index:
#            df.loc[i,promoted_out_ho] = numpy.random.permutation(df.loc[i,promoted_out_ho])
#            df.loc[i,promoted_out_hd] = numpy.random.permutation(df.loc[i,promoted_out_hd])
#            df.loc[i,promoted_out_ao] = numpy.random.permutation(df.loc[i,promoted_out_ao])
#            df.loc[i,promoted_out_ad] = numpy.random.permutation(df.loc[i,promoted_out_ad])
#            df.loc[i,relegated_out_ho] = numpy.random.permutation(df.loc[i,relegated_out_ho])
#            df.loc[i,relegated_out_hd] = numpy.random.permutation(df.loc[i,relegated_out_hd])
#            df.loc[i,relegated_out_ao] = numpy.random.permutation(df.loc[i,relegated_out_ao])
#            df.loc[i,relegated_out_ad] = numpy.random.permutation(df.loc[i,relegated_out_ad])
#        colmap = {o:i for o,i in zip(promoted_out_ho, relegated_in_ho)}
#        colmap.update({o:i for o,i in zip(promoted_out_hd, relegated_in_hd)})
#        colmap.update({o:i for o,i in zip(promoted_out_ao, relegated_in_ao)})
#        colmap.update({o:i for o,i in zip(promoted_out_ad, relegated_in_ad)})
#        colmap.update({o:i for o,i in zip(relegated_out_ho, promoted_in_ho)})
#        colmap.update({o:i for o,i in zip(relegated_out_hd, promoted_in_hd)})
#        colmap.update({o:i for o,i in zip(relegated_out_ao, promoted_in_ao)})
#        colmap.update({o:i for o,i in zip(relegated_out_ad, promoted_in_ad)})
#        df = df.rename(columns=colmap)
#        # Team offense/defense priors
#        team_names = [c[:-3] for c in df.columns if c[-3:]==' HO']
#        homeoff_pars = [t+' HO' for t in team_names]
#        homedef_pars = [t+' HD' for t in team_names]
#        awayoff_pars = [t+' AO' for t in team_names]
#        awaydef_pars = [t+' AD' for t in team_names]
#        homeoff_prior_mean = numpy.array(df[homeoff_pars].mean()) * regression
#        homeoff_prior_var = numpy.cov(df[homeoff_pars].T)
#        homedef_prior_mean = numpy.array(df[homedef_pars].mean()) * regression
#        homedef_prior_var = numpy.cov(df[homedef_pars].T)
#        awayoff_prior_mean = numpy.array(df[awayoff_pars].mean()) * regression
#        awayoff_prior_var = numpy.cov(df[awayoff_pars].T)
#        awaydef_prior_mean = numpy.array(df[awaydef_pars].mean()) * regression
#        awaydef_prior_var = numpy.cov(df[awaydef_pars].T)
#        # Assemble and return
#        return cls(homeoff_prior_mean = homeoff_prior_mean,
#                   homeoff_prior_var = homeoff_prior_var,
#                   homedef_prior_mean = homedef_prior_mean,
#                   homedef_prior_var = homedef_prior_var,
#                   awayoff_prior_mean = awayoff_prior_mean,
#                   awayoff_prior_var = awayoff_prior_var,
#                   awaydef_prior_mean = awaydef_prior_mean,
#                   awaydef_prior_var = awaydef_prior_var,
#                   team_names = team_names,
#                   log_home_goals_prior_mean = log_home_goals_prior_mean,
#                   log_home_goals_prior_sd = log_home_goals_prior_sd,
#                   log_away_goals_prior_mean = log_away_goals_prior_mean,
#                   log_away_goals_prior_sd = log_away_goals_prior_sd)