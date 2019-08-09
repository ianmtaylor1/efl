"""Module implementing a singleton pattern for a database engine. Maintains a
single sqlalchemy engine and issues connections to it for other modules. If 
the configuration dbengine has changed since last use, a new engine is 
created."""

from .. import config

import sqlalchemy

# The currently active engine
_engine = None
# The url of that engine
_url = None

# Base session maker for the module
_Sess = sqlalchemy.orm.sessionmaker()

def _get():
    """Handles the logic of checking if the configured db url has changed,
    and if so, recreating the engine."""
    global _url, _engine
    if config.dbengine != _url:
        if _engine is not None:
            _engine.dispose()
        _engine = sqlalchemy.create_engine(config.dbengine)
        _url = config.dbengine
    return _engine

def connect():
    """Issue a connection to the current database engine. If the configured
    url has changed, recreate the engine."""
    return _get().connect()

def dispose():
    """Disposes the engine's connection pool and gets a new one. Intended to
    be used only if the process using this module fork()s. More info:
    https://docs.sqlalchemy.org/en/13/core/connections.html#engine-disposal"""
    _get().dispose()

def Session():
    """Create a sqlalchemy session to the currently configured engine."""
    return _Sess(bind=_get())
