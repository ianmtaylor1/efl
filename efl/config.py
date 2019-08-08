"""Reading config file and store options to be accessed by other modules
in this package.

The objects exposed by this module are:
    parse(file) - parses the supplied file to repopulate the below options
    modelcache - directory in which to store precompiled stan models
    conffile - most recent file that was read to produce 
    dbengine - database for storage/retrieval of efl data, in 
        sqlalchemy format
"""

import configparser
import appdirs
from os import path

# DEFAULTS

_appauthor = "imtaylor"
_appname = "efl"

_default_conffile = path.join(
        appdirs.user_config_dir(appname=_appname, appauthor=_appauthor),
        'eflconf.ini'
        )

_default_modelcache = path.join(
        appdirs.user_cache_dir(appname=_appname, appauthor=_appauthor),
        'models'
        )

_default_dbengine = 'sqlite:///' + path.join(
        appdirs.user_data_dir(appname=_appname, appauthor=_appauthor),
        'efl.sqlite3'
        )

# PARSE

"""Parses a configuration file and resets the values exposed by this 
module. If the file doesn't exist, defaults are supplied instead."""
def parse(file_):
    # These are the things we will save out
    global conffile, modelcache, dbengine
    # Read in the config file
    cp = configparser.ConfigParser()
    if path.isfile(file_):
        cp.read(file_)
    # Set values
    conffile   = file_
    modelcache = cp.get('efl', 'modelcache', fallback=_default_modelcache)
    dbengine   = cp.get('efl', 'dbengine',   fallback=_default_dbengine)


######## INITIALIZATION CODE: PARSE THE DEFAULT CONFIG FILE ##########
parse(_default_conffile)


