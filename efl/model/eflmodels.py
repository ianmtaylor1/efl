# -*- coding: utf-8 -*-
"""
eflmodels.py 

Module contains code to build and sample from EFL models.
"""

from . import cache

import numpy

class _EFLModel(object):
    """Base class for EFL models. Mostly handles wrapping of the StanFit
    object, since inheritance is hard.
    
    Subclasses of EFLModel should:
        1. have a class-level attribute called _modelname
        2. Implement _inits_from_prior and _inits_from_optim"""
    
    # Name of the model (used for cache). Set in child classes.
    _modelname = None
    
    def __init__(self, modeldata, modelprior, 
                 chains=4, iter=10000, warmup=None, thin=1, n_jobs=1):
        """Initialize the base properties of this model.
        """
        # Get model from compiled model cache
        self._model = cache.get_model(self._modelname)
        # Create inits: one from optimization, rest sampled from prior
        inits = [self._inits_from_optim(modeldata, modelprior)]
        #inits = []
        for c in range(len(inits),chains):
            inits.append(self._inits_from_prior(modelprior))
        # Fit the model
        self._stanfit = self._model.sampling(
                data={**modeldata, **modelprior}, init=inits,
                chains=chains, iter=iter, warmup=warmup, thin=thin, 
                n_jobs=n_jobs)
    
    def _inits_from_optim(self, modeldata, modelprior):
        """Produce initial values for MCMC by optimizing the posterior.
        (posterior mode)"""
        raise NotImplementedError(
                "_inits_from_optim not implemented in {}".format(type(self))
                )
    
    def _inits_from_prior(self, modelprior):
        """Produce initial values for MCMC by sampling from prior."""
        raise NotImplementedError(
                "_inits_from_prior not implemented in {}".format(type(self))
                )


class _Stan_symordreg(_EFLModel):
    """Base class of all EFL models which use the symordreg.stan model."""
    
    _modelname = 'symordreg'
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def _inits_from_optim(self, modeldata, modelprior):
        """Optimize the posterior to create initial values."""
        starting = [{'beta':modelprior['beta_prior_mean'],
                    'theta':modelprior['theta_prior_loc']}]
        opt = self._model.optimizing(data={**modeldata, **modelprior},
                                     init=starting)
        return {'beta':opt['beta'], 'theta':opt['theta']}
    
    def _inits_from_prior(self, modelprior):
        """Draw from a multivariate normal distribution and a logistic
        distribution to produce prior values for beta and theta."""
        beta = numpy.random.multivariate_normal(
                modelprior['beta_prior_mean'], 
                modelprior['beta_prior_var'])
        theta = abs(numpy.random.logistic(
                modelprior['theta_prior_loc'], 
                modelprior['theta_prior_scale']))
        return {'beta':beta, 'theta':theta}


class EFLSymOrdReg(_Stan_symordreg):
    """*Sym*metric *Ord*inal *Reg*ression model for EFL data."""
    
    def __init__(self, eflgames, **kwargs):
        modeldata = self._get_model_data(eflgames)
        P = modeldata['P']
        modelprior = {'beta_prior_mean':numpy.zeros(P),
                      'beta_prior_var':numpy.identity(P) * (P**2),
                      'theta_prior_loc':0,
                      'theta_prior_scale':1}
        super().__init__(modeldata, modelprior, **kwargs)
        
    @staticmethod
    def _get_model_data(games):
        """Take an EFLGames instance and transform it into a dict appropriate for
        the symordreg Stan model."""
        N = len(games.fit)
        N_new = len(games.predict)
        P = len(games.teams)
        Y = numpy.array([2 for g in games.fit], dtype=numpy.int_)
        Y[[(g.result.homepoints > g.result.awaypoints) for g in games.fit]] = 3
        Y[[(g.result.homepoints < g.result.awaypoints) for g in games.fit]] = 1
        X = numpy.zeros(shape=[N,P])
        X[:,0] = 1 # Homefield
        X_new = numpy.zeros(shape=[N_new,P])
        X_new[:,0] = 1 # Homefield
        for ti in range(1,len(games.teams)):
            X[[g.hometeamid == games.teams[ti].id for g in games.fit], ti] = 1
            X[[g.awayteamid == games.teams[ti].id for g in games.fit], ti] = -1
            X_new[[g.hometeamid == games.teams[ti].id for g in games.predict], ti] = 1
            X_new[[g.awayteamid == games.teams[ti].id for g in games.predict], ti] = -1
        return {'N':N, 'N_new':N_new, 'P':P, 'Y':Y, 'X':X, 'X_new':X_new}    

