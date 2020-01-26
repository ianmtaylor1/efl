import pystan
import importlib.resources as resources
import os
import pickle
import argparse

from . import stanfiles
from .. import config


def get_model(model_name):
    """Get a StanModel object, checking first in local cache and recompiling
    if necessary."""
    modelfile = '{}.stan'.format(model_name)
    cachefile = os.path.join(config.modelcache, '{}.pkl'.format(model_name))
    # Check if package exists, get model code.
    if not resources.is_resource(stanfiles, modelfile):
        raise Exception(
                "Model {} is not an available Stan model.".format(model_name)
                )
    model_stanc = pystan.stanc(
            model_code    = resources.read_text(stanfiles, modelfile),
            include_paths = [os.path.join(stanfiles.__path__[0], "include")],
            model_name    = model_name
            )
    # Check if cached file exists and load it
    model = None
    if os.path.isfile(cachefile):
        try:
            with open(cachefile, 'rb') as f:
                model = pickle.load(f)
        except pickle.UnpicklingError:
            os.remove(cachefile) # Bad file, remove
    # If cache file did not exist or is from an old model, recompile and cache
    if (model is None) or (model.model_cppcode != model_stanc['cppcode']):
        model = pystan.StanModel(stanc_ret=model_stanc)
        os.makedirs(config.modelcache, exist_ok=True)
        with open(cachefile, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return model 


def clear():
    """Delete all files from the cache directory so that they will be
    recompiled the next time they are needed. Returns the list of files that
    were removed."""
    files = os.listdir(config.modelcache)
    removed = []
    for f in files:
        fullpath = os.path.join(config.modelcache, f)
        if os.path.isfile(fullpath):
            removed.append(f)
            os.remove(fullpath)
    return removed


def console_clear_cache():
    """Function to be run as a console command entry point, initiates clearing
    of compiled model cache."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Download and save EFL games.")
    parser.add_argument("--config", "-c", type=str,
            help="Configuration file to override defaults.")
    # Parse args
    args = parser.parse_args()
    # Read the optionally supplied configuration file
    if args.config is not None:
        from .. import config
        config.parse(args.config)
    # Delete cache and display which files were deleted
    print("Deleting compiled model cache:")
    for f in clear():
        print(f)

