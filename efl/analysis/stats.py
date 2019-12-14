#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats.py

Contains functions which compute statistics on sets of games (simulated or
real). Useful for the statfun argument in the ppc function.
"""

import functools
import numpy
import pandas


###########################################################
## BASE CLASS AND FUNCTION DECORATOR ######################
###########################################################


class BaseStat(object):
    """Base class for stat classes. Implements common features and provides
    a handy __init__ function."""
    
    def __init__(self, precompute, type_, name):
        self.precompute = precompute
        self.type_ = type_
        self.name = name
    
    def _id(self):
        """Returns identifying attributes of the object for hashing and
        equality comparisons. Can be a single value or a tuple of values, but
        needs to be hashable. Only needs to differentiate between members of
        the same class - class comparison is automatically handled. Should be
        equal for two instances that would result in the same computed values
        of the statistic in __call__. It's okay to leave this unimplemented,
        but it may result in lots of unnecessary repeated computation in the
        EFLPredictor."""
        raise NotImplementedError()
    
    def __call__(self, x):
        """Computes the desired statistic on the data x."""
        raise NotImplementedError()
    
    def __hash__(self):
        try:
            return hash((type(self), self._id()))
        except NotImplementedError:
            return super().__hash__()
    
    def __eq__(self, other):
        if isinstance(other, BaseStat):
            try:
                return type(self) == type(other) and self._id() == other._id()
            except NotImplementedError:
                return super().__eq__(other)
        return NotImplemented


def stat(precompute, type_, name=None, sort=None):
    """Decorator for creating stat functions for use in EFLPredictor.
    Parameters:
        precompute - what format this function is expecting to receive: ("df",
            "table" or "matrix")
        type_ - return type of this function ("numeric", "ordinal", or 
            "nominal")
        name - The name to apply to this statistic. If none, the function's
            __name__ attribute is used by EFLPredictor
        sort - if type_ is 'nominal', a key function that is used to sort
            values.
    """
    def stat_decorator(func):
        @functools.wraps(func)
        def statwrapper(x):
            return func(x)
        statwrapper.precompute = precompute
        statwrapper.type_ = type_
        if name is not None:
            statwrapper.name = name
        if sort is not None:
            statwrapper.sort = sort
        return statwrapper
    return stat_decorator


###########################################################
## DERIVED STAT CLASSES AND FUNCTIONS #####################
###########################################################


class GameResult(BaseStat):
    """This class is a stat that returns, for a given game id, what the
    result is. (Home, Draw, Away)"""
    
    def __init__(self, gameid, games=None):
        # Determine the name
        if games is None:
            name = "Result {}".format(gameid)
        else:
            gamedf = games.to_dataframe(fit=True, predict=True)
            hometeam = gamedf.loc[gameid, 'hometeam']
            awayteam = gamedf.loc[gameid, 'awayteam']
            name = '{} vs {} Result'.format(hometeam, awayteam)
        # Construct the BaseStat
        super().__init__('df', 'ordinal', name)
        # Save the game id
        self._gameid = gameid
    
    def _id(self):
        return self._gameid
    
    def __call__(self, df):
        return df.loc[self._gameid, 'result']
    
    @staticmethod
    def sort(v):
        # Sorts in the order H,D,A
        return {'H':1, 'D':2, 'A':3}.get(v, 4)


class GameScore(BaseStat):
    """This class is a stat that returns, for a given game id, what the
    score is as a tuple: (homegoals, awaygoals)"""
    
    def __init__(self, gameid, games=None):
        # Determine the name
        if games is None:
            name = "Score {}".format(gameid)
        else:
            gamedf = games.to_dataframe(fit=True, predict=True)
            hometeam = gamedf.loc[gameid, 'hometeam']
            awayteam = gamedf.loc[gameid, 'awayteam']
            name = '{} vs {} Score'.format(hometeam, awayteam)
        # Construct the BaseStat
        super().__init__('df', 'ordinal', name)
        # Save the game id
        self._gameid = gameid
    
    def _id(self):
        return self._gameid
    
    def __call__(self, df):
        return (df.loc[self._gameid,'homegoals'], df.loc[self._gameid,'awaygoals'])
    
    @staticmethod
    def sort(v):
        # Sorts from largest home win to largest away win
        margin = v[0] - v[1]
        if margin >= 0:
            total = v[0] + v[1]
        else:
            total = -(v[0] + v[1])
        return (-margin, -total)


class GameMargin(BaseStat):
    """This class is a stat that returns, for a given game id, what the
    margin is: homegoals - awaygoals"""
    
    def __init__(self, gameid, games=None):
        # Determine the name
        if games is None:
            name = "Margin {}".format(gameid)
        else:
            gamedf = games.to_dataframe(fit=True, predict=True)
            hometeam = gamedf.loc[gameid, 'hometeam']
            awayteam = gamedf.loc[gameid, 'awayteam']
            name = '{} vs {} Margin'.format(hometeam, awayteam)
        # Construct the BaseStat
        super().__init__('df', 'ordinal', name)
        # Save the game id
        self._gameid = gameid
    
    def _id(self):
        return self._gameid
    
    def __call__(self, df):
        return df.loc[self._gameid,'homegoals'] - df.loc[self._gameid,'awaygoals']
    
    @staticmethod
    def sort(v):
        # Sorts from largest home win to largest away win
        return -v


class GameTotalGoals(BaseStat):
    """This class is a stat that returns, for a given game id, what the
    total score is: homegoals + awaygoals"""
    
    def __init__(self, gameid, games=None):
        # Determine the name
        if games is None:
            name = "Total Goals {}".format(gameid)
        else:
            gamedf = games.to_dataframe(fit=True, predict=True)
            hometeam = gamedf.loc[gameid, 'hometeam']
            awayteam = gamedf.loc[gameid, 'awayteam']
            name = '{} vs {} Total Goals'.format(hometeam, awayteam)
        # Construct the BaseStat
        super().__init__('df', 'ordinal', name)
        # Save the game id
        self._gameid = gameid
    
    def _id(self):
        return self._gameid
    
    def __call__(self, df):
        # This assumes there is only one row for every gameid
        return df.loc[self._gameid,'homegoals'] + df.loc[self._gameid,'awaygoals']


class LeaguePosition(BaseStat):
    """This class returns the team that is in the supplied position within 
    the league. (1 = Winner, 2 = runner-up, etc). Issues: Ties (points, goal
    difference, and goals for all equal) are arbitrarily decided. In real
    tables, teams share a rank."""
    
    def __init__(self, position):
        super().__init__('table', 'nominal',
                         "League Position {}".format(position))
        self._pos = position

    def _id(self):
        return self._pos
    
    def __call__(self, t):
        points = t['HPts'] + t['APts']
        gf = (t['HGF'] + t['AGF']).fillna(0)
        gd = (gf - t['HGA'] - t['AGA']).fillna(0)
        t2 = pandas.DataFrame({'pts':points, 'gd':gd, 'gf':gf,
                               'rnd':numpy.random.normal(size=len(points))})
        return t2.sort_values(['pts','gd','gf','rnd'],ascending=False).index[self._pos-1]


class TeamPosition(BaseStat):
    """This class returns the position within the league of the specified
    team. (1=Winner, 2=Runner-up, etc) Issues: Ties (points, goal
    difference, and goals for all equal) are arbitrarily decided. In real
    tables, teams share a rank."""
    
    def __init__(self, team):
        super().__init__('table', 'ordinal', "{} Position".format(team))
        self._team = team
        
    def _id(self):
        return self._team
    
    def __call__(self, t):
        points = t['HPts'] + t['APts']
        gf = (t['HGF'] + t['AGF']).fillna(0)
        gd = (gf - t['HGA'] - t['AGA']).fillna(0)
        t2 = pandas.DataFrame({'pts':points, 'gd':gd, 'gf':gf,
                               'rnd':numpy.random.normal(size=len(points))})
        t2 = t2.sort_values(['pts','gd','gf','rnd'], ascending=False)
        t2['rank'] = range(1,len(t2)+1)
        return t2.loc[self._team,'rank']


class TeamPoints(BaseStat):
    """This class is a stat that returns, for a given team, how many points
    the team has total (Points are 3 for a win, 1 for a draw.)"""
    
    def __init__(self, team):
        super().__init__('table', 'numeric', '{} points'.format(team))
        self._team = team

    def _id(self):
        return self._team
    
    def __call__(self, t):
        return t.loc[self._team,'HPts'] + t.loc[self._team,'APts']


@stat('df','numeric')
def homewinpct(g):
    """Calculates the homefield advantage: what percentage of games were won
    by the home team."""
    return 100*sum(g['result'] == 'H')/g['result'].count()


@stat('df','numeric')
def drawpct(g):
    """Calculates the likelihood of a draw: what percentage of games resulted
    in a draw."""
    return 100*sum(g['result'] == 'D')/g['result'].count()


@stat('table','numeric')
def maxpoints(t):
    """Calculates the max points by a team. (Points are 3 for a win, 1 for a 
    draw.)"""
    return numpy.max(t['HPts'] + t['APts'])


@stat('table','numeric')
def minpoints(t):
    """Calculates the min points by a team. (Points are 3 for a win, 1 for a
    draw.)"""
    return numpy.min(t['HPts'] + t['APts'])


@stat('table','numeric')
def sdpoints(t):
    """Calculates the standard deviation of points by a team. (Points are 3
    for a win, 1 for a draw.)"""
    return numpy.std(t['HPts'] + t['APts'])

@stat('table','numeric')
def sdhomeadv(t):
    """Calculates the standard deviation of the homefield advantage for all
    teams. Homefield advantage is defined as home points minus away points."""
    return numpy.std(t['HPts'] - t['APts'])


@stat('table','numeric')
def maxhomeadv(t):
    """Calculates the maximum homefield advantage for all teams. Homefield
    advantage is defined as home points minus away points."""
    return numpy.max(t['HPts'] - t['APts'])

@stat('matrix','numeric')
def numrecip(M):
    """Calculates the number of reciprocal game pairs. (e.g. Team A beats
    Team B, then Team B beats Team A.)"""
    M2 = numpy.matmul(M, M)
    return M2.diagonal().sum() / 2


@stat('matrix','numeric')
def numtriangles(M):
    """Calculates the number of 'triangles' (e.g. A > B > C > A)"""
    M3 = numpy.matmul(numpy.matmul(M, M), M)
    return M3.diagonal().sum() / 3


@stat('df','numeric',name='Goals Index of Dispersion')
def goals_ind_disp(df):
    """Calculates the index of dispersion (var/mean) for all goals scored in
    all games."""
    allgoals = df['homegoals'].append(df['awaygoals'])
    return allgoals.var()/allgoals.mean()

@stat('df','numeric',name='Goals Coefficient of Variation')
def goals_cv(df):
    """Calculates the coefficient of variation (sd/mean) for all goals scored
    in all games."""
    allgoals = df['homegoals'].append(df['awaygoals'])
    return allgoals.std()/allgoals.mean()


@stat('df','numeric')
def avghomemargin(df):
    """Calculates the average home team margin of victory for all games."""
    return (df['homegoals']-df['awaygoals']).mean()

