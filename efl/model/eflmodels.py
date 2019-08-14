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
    
    This class provides/creates:
        1. An __init__ method that builds and fits the model
        2. Instance attribute: _model 
        3. Instance attribute: stanfit
        4. Instance attribute: _modeldata (equal to whatever was passed to
                __init__). Useful for _stan_inits function
    
    Subclasses of EFLModel should:
        1. have a class-level attribute called _modelfile
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
        self.stanfit = self._model.sampling(
                data=self._modeldata, init=self._stan_inits,
                chains=chains, iter=iter, warmup=warmup, thin=thin, 
                n_jobs=n_jobs)
    
    def _stan_inits(self, chain_id=None):
        """Produce initial values for MCMC. Should return a dict with keys
        equal to Stan model parameters, and values equal to their initial
        values."""
        raise NotImplementedError(
                "_stan_inits not implemented in {}".format(type(self))
                )
    
    def summary(self, pars=None, **kwargs):
        """A wrapper around the stansummary method on the included stanfit
        object. It will convert parameter names as defined in _stan2efl, and
        by default will only include those parameters which are keys in 
        that dict."""
        # Fill default pars and map to stan names
        if pars is None:
            pars = self._efl2stan.keys()
        stspars = [self._efl2stan[p] for p in pars]
        # Run stansummary for the underlying fit
        sts = self.stanfit.stansummary(pars=stspars, **kwargs)
        # Translate summary to useful parameter names
        addlength = max(len(p) for p in self._stan2efl.values()) \
                    - max(len(p) for p in self._stan2efl.keys())
        for (stanpar, eflpar) in self._stan2efl.items():
            spaces = addlength - (len(eflpar) - len(stanpar))
            if spaces >= 0: # Need to net insert spaces
                old = stanpar
                new = '{}{}'.format(eflpar, " "*spaces)
            else: # Need to net remove spaces
                old = '{}{}'.format(stanpar, " "*abs(spaces))
                new = eflpar
            sts = sts.replace(old, new)
        # Also add spaces at the start of the header row
        if addlength > 0:
            sts = sts.replace(" mean ", "{} mean ".format(" "*addlength))
        elif addlength < 0:
            sts = sts.replace("{} mean ".format(" "*abs(addlength)), " mean ")
        return sts
    
    def to_dataframe(self, pars=None, **kwargs):
        """A wrapper around the to_dataframe method on the included stanfit
        object. It will convert column par names as defined in _stan2efl, and
        by default will only include those parameters which are keys in 
        that dict."""
        # Fill default pars and map to stan names
        if pars is None:
            pars = self._efl2stan.keys()
        stspars = [self._efl2stan[p] for p in pars]
        # Run to_dataframe for the underlying fit
        df = self.stanfit.to_dataframe(pars=stspars, **kwargs)
        # Translate the column names to useful parameter names
        df.columns = [self._stan2efl.get(c, c) for c in df.columns]
        return df


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
    
    
    def __init__(self, eflgames, **kwargs):
        modeldata, self._reference = self._get_model_data(eflgames)
        # TODO: get prior from previous fit or another way
        P = modeldata['P']
        modeldata['beta_prior_mean'] = numpy.zeros(P)
        modeldata['beta_prior_var'] = numpy.identity(P) * ((P/2)**2)
        modeldata['theta_prior_loc'] = 0
        modeldata['theta_prior_scale'] = 1
        # Call the superclass to fit the model
        super().__init__(modeldata, **kwargs)
        # Create parameter mappings.
        self._efl2stan = {'DrawBnd':'theta', 'HomeField':'beta[1]'}
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
        Y[[(g.result.homegoals > g.result.awaygoals) for g in games.fit]] = 3
        Y[[(g.result.homegoals < g.result.awaygoals) for g in games.fit]] = 1
        X = numpy.zeros(shape=[N,P])
        X[:,0] = 1 # Homefield
        X_new = numpy.zeros(shape=[N_new,P])
        X_new[:,0] = 1 # Homefield
        for ti in range(1,len(games.teams)):
            X[[g.hometeamid == games.teams[ti].id for g in games.fit], ti] = 1
            X[[g.awayteamid == games.teams[ti].id for g in games.fit], ti] = -1
            X_new[[g.hometeamid == games.teams[ti].id for g in games.predict], ti] = 1
            X_new[[g.awayteamid == games.teams[ti].id for g in games.predict], ti] = -1
        return {'N':N, 'N_new':N_new, 'P':P, 'Y':Y, 'X':X, 'X_new':X_new}, games.teams[0].shortname
    
    def summary(self, pars=None, **kwargs):
        """Decorate the default summary. If pars is left as default, or
        includes the reference team, the reference is printed below all other 
        parameters."""
        if (pars is None) or (self._reference in pars):
            addref = True
        else:
            addref = False
        sts = super().summary(pars, **kwargs)
        if addref:
            newline = '**Reference: {} = 0'.format(self._reference)
            lines = sts.split("\n")
            # What is the last line that contains the name of a parameter?
            haspar = [any((p in ln) for p in self._efl2stan.keys()) for ln in lines]
            lastpar = max(i for i in range(len(haspar)) if haspar[i])
            # Inser the holdout right after that.
            lines.insert(lastpar + 1, newline)
            sts = "\n".join(lines)
        return sts
    
    def to_dataframe(self, pars=None, **kwargs):
        """Decorate the default to_dataframe. If pars is left as default, or
        includes the reference team, the reference is added as a column of
        all zeros."""
        if (pars is None) or (self._reference in pars):
            addref = True
        else:
            addref = False
        df = super().to_dataframe(pars, **kwargs)
        if addref:
            ispar = [(c in self._efl2stan.keys()) for c in df.columns]
            lastpar = max(i for i in range(len(ispar)) if ispar[i])
            df.insert(lastpar + 1, column=self._reference, value=0)
        return df
