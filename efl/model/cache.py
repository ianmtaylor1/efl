import pystan
import importlib.resources as resources
import appdirs
import os
import pickle
from . import stanfiles

_cache_appname = "efl"
_cache_appauthor = "imtaylor"
_cache_subdir = "models"

"""Get a StanModel object, checking first in local cache and recompiling
if necessary"""
def get_model(model_name):
    cachedir = appdirs.user_cache_dir(_cache_appname, _cache_appauthor)
    modelfile = '{}.stan'.format(model_name)
    cachefile = os.path.join(
            cachedir, _cache_subdir, '{}.pkl'.format(model_name)
            )
    # Check if package exists, get model code.
    if not resources.is_resource(stanfiles, modelfile):
        raise Exception(
                "Model {} is not an available Stan model.".format(model_name)
                )
    modelcode = resources.read_text(stanfiles, modelfile)
    # Check if cached file exists and load it
    model = None
    if os.path.isfile(cachefile):
        try:
            with open(cachefile, 'rb') as f:
                model = pickle.load(f)
        except pickle.UnpicklingError as e:
            os.remove(cachefile) # Bad file, remove
    # If cache file did not exist or is from an old model, recompile and cache
    if (model is None) or (model.model_code != modelcode):
        model = pystan.StanModel(model_code=modelcode, model_name=model_name)
        os.makedirs(os.path.dirname(cachefile), exist_ok=True)
        with open(cachefile, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return(model)


