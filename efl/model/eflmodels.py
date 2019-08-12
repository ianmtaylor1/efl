# -*- coding: utf-8 -*-
"""
eflmodels.py 

Module contains code to build and sample from EFL models.
"""

from . import cache

import numpy


def _get_symordreg_data(games, holdout=None):
    """Take an EFLGames instance and transform it into a dict appropriate for
    the symordreg Stan model."""
    # 
    N = len(games.fit)
    N_new = len(games.predict)
    P = len(games.teams)
    Y = numpy.array([2 for g in games.fit], dtype=numpy.int_)
    Y[[(g.result.homepoints > g.result.awaypoints) for g in games]] = 3
    Y[[(g.result.homepoints < g.result.awaypoints) for g in games]] = 1
    X = numpy.zeros(shape=[N,P])
    for row in range(len(games.fit)):
        X[row,games.teams.index(games.fit.hometeam)] = 1
        X[row,games.teams.index(games.fit.awayteam)] = -1
    X_new = numpy.zeros(shape=[N_new,P])
    for row in range(len(games.predict)):
        X_new[row,games.teams.index(games.predict.hometeam)] = 1
        X_new[row,games.teams.index(games.predict.awayteam)] = -1
    return {'N':N, 'N_new':N_new, 'P':P, 'Y':Y, 'X':X, 'X_new':X_new}    

def build_symordreg(games):
    model = cache.get_model("symordreg")
    