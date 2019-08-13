# -*- coding: utf-8 -*-
"""
eflmodels.py 

Module contains code to build and sample from EFL models.
"""

from . import cache

import numpy
import pystan

class _EFLModel(object):
    """Base class for EFL models. Mostly handles wrapping of the StanFit
    object, since inheritance is hard.
    
    This class provides/creates:
        1. An __init__ method that builds and fits the model
        2. Instance attributes: _model and _stanfit
        3. Instance attributes: _modeldata (equal to whatever was passed to
                __init__). May be used in _stan_inits function
    
    Subclasses of EFLModel should:
        1. have a class-level attribute called _modelfile
        2. Have an instance attribute called model_name
        2. Implement _stan_inits method for generating initial values 
                from chain_id
        3. Have object-level attributes _efl2stan and _stan2efl for parameter
                name mapping between Stan output and EFL-relevant names.
    """
    
    def __init__(self, modeldata,
                 chains=4, iter=10000, warmup=None, thin=1, n_jobs=1):
        """Initialize the base properties of this model.
        Parameters:
            modeldata - Data that is passed to sampling(data=modeldata). Is
                also stored as self._modeldata for use in other methods.
            chains, iter, warmup, thin, n_jobs - same as pystan options.
                Passed to sampling()
        """
        # Get model from compiled model cache
        self._model = cache.get_model(self._modelfile)
        # Store the data that was passed as an instance attribute
        self._modeldata = modeldata
        # Fit the model
        self._stanfit = self._model.sampling(
                data=self._modeldata, init=self._stan_inits,
                chains=chains, iter=iter, warmup=warmup, thin=thin, 
                n_jobs=n_jobs)
    
    def _stan_inits(self, chain_id=None):
        """Produce initial values for MCMC. Should return a dict with keys
        equal to Stan model parameters, and values equal to their initial
        values."""
        raise NotImplementedError(
                "_inits_from_optim not implemented in {}".format(type(self))
                )
    
    ######## Read only property passed through to _stanfit
    ######## Required to trick stansummary into using this object.
    @property
    def sim(self):
        return self._stanfit.sim
    
    ######## Read only property passed through to _stanfit
    ######## Required to trick stansummary into using this object.
    @property
    def mode(self):
        return self._stanfit.mode
    
    ######## Read only property passed through to _stanfit
    ######## Required to trick stansummary into using this object.
    @property
    def date(self):
        return self._stanfit.date
    
    def summary(self, pars=None, probs=None):
        """Return a summary in the same format as the pystan fit summary.
        Maps parameter names to the ones defined by the model, instead of
        the default stan ones."""
        if pars is None:
            pars = self._efl2stan.keys()
        # Map the desired parameters to their Stan equivalents
        stanpars = [self._efl2stan.get(p, p) for p in pars]
        # Get the Stan summary
        summ = self._stanfit.summary(pars=stanpars, probs=probs)
        # Remap the parameter names to their EFL equivalents
        summ['summary_rownames'] = [self._stan2efl.get(p, p) for p in summ['summary_rownames']]
        summ['c_summary_rownames'] = [self._stan2efl.get(p, p) for p in summ['c_summary_rownames']]
        return summ
    
    def stansummary(self, **kwargs):
        """Make a Stan Summary for this object. See: pystan.stansummary.
        Arguments: keyword arguments identical to pystan.stansummary()
        
        Note: This works by implementing the following methods/attributes:
            * summary() (with remapped parameter names)
            * sim
            * mode
            * date
        If pystan.stansummary is changed to use different attributes, more
        changes will be needed to this base class.
        """
        return pystan.stansummary(self, **kwargs)


class _Stan_symordreg(_EFLModel):
    """Base class of all EFL models which use the symordreg.stan model."""
    
    _modelfile = 'symordreg'
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    
    def _stan_inits(self, chain_id=None):
        """Draw from a multivariate normal distribution and a logistic
        distribution to produce prior values for beta and theta."""
        beta = numpy.random.multivariate_normal(
                self._modeldata['beta_prior_mean'], 
                self._modeldata['beta_prior_var'])
        theta = abs(numpy.random.logistic(
                self._modeldata['theta_prior_loc'], 
                self._modeldata['theta_prior_scale']))
        return {'beta':beta, 'theta':theta}


class EFLSymOrdReg(_Stan_symordreg):
    """*Sym*metric *Ord*inal *Reg*ression model for EFL data."""
    
    model_name = 'EFLSymOrdReg'
    
    def __init__(self, eflgames, **kwargs):
        modeldata = self._get_model_data(eflgames)
        # TODO: get prior from previous fit or another way
        P = modeldata['P']
        modeldata['beta_prior_mean'] = numpy.zeros(P)
        modeldata['beta_prior_var'] = numpy.identity(P) * (P**2)
        modeldata['theta_prior_loc'] = 0
        modeldata['theta_prior_scale'] = 1
        # Call the superclass to fit the model
        super().__init__(modeldata, **kwargs)
        # Create parameter mappings.
        self._efl2stan = {'DrawThreshold':'theta', 'HomeAdvantage':'beta[1]'}
        for ti in range(1, len(eflgames.teams)):
            self._efl2stan[eflgames.teams[ti].shortname] = 'beta[{}]'.format(ti+1)
        self._stan2efl = dict(reversed(i) for i in self._efl2stan.items())
        
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

