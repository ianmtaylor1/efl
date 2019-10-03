#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats.py

Contains functions which compute statistics on sets of games (simulated or
real). Useful for the statfun argument in the ppc function.
"""

import pandas
import numpy

def _make_table(g):
    """Take the data frame of games and transform it into a league table.
    Columns in the resulting table follow the format (H|A)(GF|GA|Pts) for
    (home/away) (goals for/goals against/points)."""
    # Pre-group for efficiency
    grp_home = g.groupby('hometeam')
    grp_away = g.groupby('awayteam')
    # Home Goals For, Home Goals Against, Away Goals For, Away Goals Against
    hgf = grp_home['homegoals'].sum()
    hga = grp_home['awaygoals'].sum()
    agf = grp_away['awaygoals'].sum()
    aga = grp_away['homegoals'].sum()
    # Home Wins, Home Draws, Home Wins, Home Draws
    hw = grp_home['result'].agg(lambda x: sum(x == 'H'))
    hd = grp_home['result'].agg(lambda x: sum(x == 'D'))
    aw = grp_away['result'].agg(lambda x: sum(x == 'A'))
    ad = grp_away['result'].agg(lambda x: sum(x == 'D'))
    # Build table
    table = pandas.DataFrame({
            'HGF':hgf, 'HGA':hga,
            'AGF':agf, 'AGA':aga,
            'HPts':3*hw + hd,
            'APts':3*aw + ad})
    return table

def _make_matrix(g):
    """Take the data frame of games and transform it into a matrix representing
    a directed multigraph of wins. E.g. if team 1 beat team 5 twice, then
    M[1,5] = 2. Draws are not included in the matrix."""
    teams = list(set(g['hometeam']) | set(g['hometeam']))
    teamidx = {t:i for i,t in enumerate(teams)}
    P = len(teams)
    mat = numpy.zeros(shape=[P,P])
    for i,row in g.iterrows():
        if row['result'] == 'H':
            mat[teamidx[row['hometeam']],teamidx[row['awayteam']]] += 1
        elif row['result'] == 'A':
            mat[teamidx[row['awayteam']],teamidx[row['hometeam']]] += 1
    return mat

def homewins(g):
    """Calculates the homefield advantage: what proportion of games in the
    DataFrame g were won by the home team."""
    return sum(g['result'] == 'H')/g['result'].count()

def draws(g):
    """Calculates the likelihood of a draw: what proportion of games in the
    DataFrame g resulted in a draw."""
    return sum(g['result'] == 'D')/g['result'].count()

def maxpoints(g):
    """Calculates the max points by a team, based on the set of games. (Points
    are 3 for a win, 1 for a draw.)"""
    t = _make_table(g)
    return numpy.max(t['HPts'] + t['APts'])

def minpoints(g):
    """Calculates the min points by a team, based on the set of games. (Points
    are 3 for a win, 1 for a draw.)"""
    t = _make_table(g)
    return numpy.min(t['HPts'] + t['APts'])

def sdpoints(g):
    """Calculates the standard deviation of points by a team, based on the set
    of games. (Points are 3 for a win, 1 for a draw.)"""
    t = _make_table(g)
    return numpy.std(t['HPts'] + t['APts'])

def sdhomeadv(g):
    """Calculates the standard deviation of the homefield advantage for all
    teams in this set of games. Homefield advantage is defined as home points
    minus away points."""
    t = _make_table(g)
    return numpy.std(t['HPts'] - t['APts'])

def maxhomeadv(g):
    """Calculates the maximum homefield advantage for all teams in this set of
    games. Homefield advantage is defined as home points minus away points."""
    t = _make_table(g)
    return numpy.max(t['HPts'] - t['APts'])

def numrecip(g):
    """Calculates the number of reciprocal game pairs. (e.g. Team A beats
    Team B, then Team B beats Team A.)"""
    M = _make_matrix(g)
    M2 = numpy.matmul(M, M)
    return M2.diagonal().sum() / 2

def numtriangles(g):
    """Calculates the number of 'triangles' (e.g. A > B > C > A)"""
    M = _make_matrix(g)
    M3 = numpy.matmul(numpy.matmul(M, M), M)
    return M3.diagonal().sum() / 3
