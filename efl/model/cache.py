import pystan
import importlib.resources as resources
import contextlib
import tempfile
import os
import pickle
import argparse

from . import stanfiles
from .. import config

@contextlib.contextmanager
def _include_path():
    """Extracts all includes into a temporary directory for use as
    include_paths in pystan.stanc. yeilds the absolute path of the directory
    in which they are (temporarily) stored."""
    # Create temporary directory. Context manager here handles cleanup
    with tempfile.TemporaryDirectory() as tempdir:
        # Copy all resources from stanfiles.includes to temporary directory
        for r in resources.contents(stanfiles.includes):
            if resources.is_resource(stanfiles.includes, r):
                with open(os.path.join(tempdir, r),'w') as f:
                    f.write(resources.read_text(stanfiles.includes, r))
        yield tempdir
        # The context manager of tempdir handles cleanup/exceptions

@contextlib.contextmanager
def _model_file(name):
    """Extracts the desired model file into a temporary directory for use as
    the model file argument to pystan.stanc. yields the absolute path of the
    file."""
    # Create temporary directory. Context manager here handles cleanup
    with tempfile.TemporaryDirectory() as tempdir:
        modelfile = '{}.stan'.format(name)
        if not resources.is_resource(stanfiles, modelfile):
            raise Exception(
                    "Model {} is not an available Stan model.".format(name)
                    )
        with open(os.path.join(tempdir, modelfile),'w') as f:
            f.write(resources.read_text(stanfiles, modelfile))
        yield os.path.join(tempdir, modelfile)
        # The context manager of tempdir handles cleanup/exceptions

def get_model(model_name):
    """Get a StanModel object, checking first in local cache and recompiling
    if necessary."""
    cachefile = os.path.join(config.modelcache, '{}.pkl'.format(model_name))
    # Check if package exists, get model code.
    with _include_path() as incpth, _model_file(model_name) as modelfile:
        model_stanc = pystan.stanc(
                file          = modelfile,
                include_paths = [incpth],
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

